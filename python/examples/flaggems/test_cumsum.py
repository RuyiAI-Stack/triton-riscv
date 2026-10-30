import pytest
import torch

from .cumsum import (
    cumsum,
    cumsum_out,
    normed_cumsum,
    reduce_then_scan_block_scan_kernel_row,
)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (1024,)])
@pytest.mark.parametrize("dim", [0, -1])
def test_cumsum(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = torch.cumsum(x, dim=dim)
    tri_out = cumsum(x, dim=dim)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(4, 32)])
@pytest.mark.parametrize("dim", [0, 1])
def test_cumsum_int(shape, dim):
    torch.manual_seed(0)
    x = torch.randint(1, 10, shape, device="cpu", dtype=torch.int32)

    ref_out = torch.cumsum(x, dim=dim)
    tri_out = cumsum(x, dim=dim)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(4, 8, 16)])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_cumsum_3d(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = torch.cumsum(x, dim=dim)
    tri_out = cumsum(x, dim=dim)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(4, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_cumsum_dtype(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=dtype)

    ref_out = torch.cumsum(x, dim=0)
    tri_out = cumsum(x, dim=0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


def test_cumsum_out():
    torch.manual_seed(0)
    x = torch.randn(64, device="cpu", dtype=torch.float32)
    out = torch.empty(64, device="cpu", dtype=torch.float32)

    ref_out = torch.cumsum(x, dim=0)
    tri_out = cumsum_out(x, dim=0, out=out)

    assert tri_out.data_ptr() == out.data_ptr(), "cumsum_out should write to out"
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_cumsum_dtype_param():
    torch.manual_seed(0)
    x = torch.randint(1, 10, (32,), device="cpu", dtype=torch.int32)

    ref_out = torch.cumsum(x, dim=0).to(torch.float32)
    tri_out = cumsum(x, dim=0, dtype=torch.float32)

    assert tri_out.dtype == torch.float32
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_cumsum_large():
    """Test large N that triggers reduce_then_scan_row (>16384)."""
    torch.manual_seed(0)
    N = 20000
    x = torch.randn(N, device="cpu", dtype=torch.float32)

    ref_out = torch.cumsum(x, dim=0)
    tri_out = cumsum(x, dim=0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_cumsum_empty():
    x = torch.empty(0, device="cpu", dtype=torch.float32)

    ref_out = torch.cumsum(x, dim=0)
    tri_out = cumsum(x, dim=0)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(4, 16)])
def test_normed_cumsum(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref = torch.div(torch.cumsum(x, dim=-1), torch.sum(x, dim=-1, keepdim=True))
    tri = normed_cumsum(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("dim", [-1, 0])
def test_cumsum_scalar(dtype, dim):
    x = torch.tensor(3, dtype=dtype)
    tri_out = cumsum(x, dim)
    ref_out = torch.cumsum(x, dim)
    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize(
    "size,dtype",
    [
        (4090, torch.bool),
        (16384, torch.bool),
        (16385, torch.bool),
        (16385, torch.int32),
        (40000, torch.int64),
        (4090, torch.float64),
        (16384, torch.float64),
        (16385, torch.float64),
    ],
)
def test_cumsum_block_prefix_boundaries(size, dtype):
    positions = torch.arange(size)
    if dtype == torch.bool:
        x = positions.remainder(11).eq(0)
    elif dtype.is_floating_point:
        x = positions.to(dtype).remainder(11) / 8 + 2.0**-30
    else:
        # Beyond float32 exact integers, but no int64 overflow.
        x = positions.remainder(11).to(dtype) + 2**24 + 1
    tri_out = cumsum(x, dim=0)
    ref_out = torch.cumsum(x, dim=0)
    torch.testing.assert_close(tri_out, ref_out, rtol=0, atol=0)


@pytest.mark.parametrize("size", [16384, 16385])
@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.uint8])
def test_cumsum_narrow_integer_block_prefix_boundaries(size, dtype):
    # Cross the row-kernel dispatch boundary and overflow every output dtype.
    x = torch.full((size,), 3, dtype=dtype)
    ref_out = torch.cumsum(x, 0, dtype=dtype)
    tri_out = cumsum(x, 0, dtype=dtype)
    torch.testing.assert_close(tri_out, ref_out, rtol=0, atol=0)

    # out= infers the narrow output dtype without an explicit dtype argument.
    out = torch.empty_like(x)
    assert cumsum_out(x, 0, out=out) is out
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_cumsum_block_prefix_mask_preserves_first_block(dtype):
    x = torch.arange(1, 14, dtype=dtype)
    # The false-mask address points to a valid, nonzero sentinel. The second
    # block reads the first block's inclusive total, exercising the true mask.
    prefix_storage = torch.tensor([49, 36, 91], dtype=dtype)
    tri_out = torch.empty_like(x)
    reduce_then_scan_block_scan_kernel_row[(1, 2, 1)](
        x,
        prefix_storage[1:],
        tri_out,
        x.numel(),
        2,
        1,
        TILE_SIZE=8,
    )
    torch.testing.assert_close(tri_out, torch.cumsum(x, 0), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.float64])
@pytest.mark.parametrize("dim", [0, 1])
def test_cumsum_strided_column_precision(dtype, dim):
    # Exercise scan_then_fan's recursively scanned block totals.
    x = torch.arange(15000).reshape(5000, 3)
    if dtype == torch.int32:
        x = x.remainder(7).to(dtype) + 2**24 + 1
    else:
        x = x.to(dtype).remainder(7) / 8 + 2.0**-30
    if dim == 1:
        x = x.T
    tri_out = cumsum(x, dim)
    ref_out = torch.cumsum(x, dim)
    torch.testing.assert_close(tri_out, ref_out, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.uint8,
        torch.int32,
        torch.int64,
    ],
)
def test_cumsum_explicit_dtype_converts_before_scan(dtype):
    if dtype in (torch.int8, torch.int16, torch.uint8):
        x = torch.tensor([1.75, 1.25, 2.5, 3.75], dtype=torch.float64)
    else:
        x = torch.tensor([1.75, -1.25, 2.5, 2**24 + 1], dtype=torch.float64)
    tri_out = cumsum(x, 0, dtype=dtype)
    ref_out = torch.cumsum(x, 0, dtype=dtype)
    torch.testing.assert_close(tri_out, ref_out, rtol=0, atol=0)


def test_cumsum_out_strided_dtype_and_alias():
    x = torch.arange(20000, dtype=torch.int64).reshape(2, 10000).remainder(11)
    ref_out = torch.cumsum(x, -1, dtype=torch.float64)
    out = torch.empty(10000, 2, dtype=torch.float64).T
    result = cumsum_out(x, -1, out=out)
    assert result is out
    torch.testing.assert_close(result, ref_out, rtol=0, atol=0)
    expected_alias = torch.cumsum(x, -1)
    assert cumsum_out(x, -1, out=x) is x
    torch.testing.assert_close(x, expected_alias, rtol=0, atol=0)


def test_cumsum_large_integer_rows():
    x = torch.arange(40000, dtype=torch.int64).reshape(2, 20000).remainder(11)
    ref_out = torch.cumsum(x, -1)
    tri_out = cumsum(x, -1)
    torch.testing.assert_close(tri_out, ref_out, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_cumsum_preserves_existing_default_dimension(dtype):
    input = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    ref_out = torch.cumsum(input, 0)
    tri_out = cumsum(input)
    torch.testing.assert_close(tri_out, ref_out)
    out = torch.empty_like(ref_out)
    assert cumsum_out(input, out=out) is out
    torch.testing.assert_close(out, ref_out)
    tri_out = cumsum(input, dim=0)
    ref_out = torch.cumsum(input, 0)
    torch.testing.assert_close(tri_out, ref_out)


def test_cumsum_scalar_out_dtype():
    input = torch.tensor(2.5)
    out = torch.empty((), dtype=torch.float64)
    assert cumsum_out(input, 0, out=out) is out
    torch.testing.assert_close(out, torch.cumsum(input, 0, dtype=torch.float64))
