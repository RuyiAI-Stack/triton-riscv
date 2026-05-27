import pytest
import torch

from .cumsum import cumsum, cumsum_out, normed_cumsum


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

    assert tri_out.data_ptr() == out.data_ptr(), (
        "cumsum_out should write to out"
    )
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

    ref = torch.cumsum(x, dim=-1) / torch.sum(x, dim=-1, keepdim=True)
    tri = normed_cumsum(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
