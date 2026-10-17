import pytest
import torch

from .nanmedian import nanmedian, nanmedian_dim, nanmedian_dim_values, nanmedian_out


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nanmedian_flat(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    mask = torch.rand(shape, device="cpu") < 0.2
    x[mask] = float("nan")

    tri_out = nanmedian(x)
    ref_out = torch.nanmedian(x)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nanmedian_dim(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    mask = torch.rand(shape, device="cpu") < 0.2
    x[mask] = float("nan")

    dim = len(shape) - 1

    tri_out = nanmedian_dim(x, dim=dim, keepdim=True)
    ref_out = torch.nanmedian(x, dim=dim, keepdim=True)

    torch.testing.assert_close(tri_out.values, ref_out.values)
    torch.testing.assert_close(tri_out.indices, ref_out.indices)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nanmedian_out(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    mask = torch.rand(shape, device="cpu") < 0.2
    x[mask] = float("nan")

    out = torch.empty((), dtype=dtype, device="cpu")

    ret = nanmedian_out(x, out=out)
    ref_out = torch.nanmedian(x)

    assert ret is out
    ref_val = ref_out[0] if isinstance(ref_out, tuple) else ref_out
    torch.testing.assert_close(out, ref_val)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nanmedian_dim_values(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    mask = torch.rand(shape, device="cpu") < 0.2
    x[mask] = float("nan")

    dim = len(shape) - 1

    out_shape = list(shape)
    out_shape.pop(dim)
    values = torch.empty(out_shape, device="cpu", dtype=dtype)
    indices = torch.empty(out_shape, device="cpu", dtype=torch.int64)

    _ = nanmedian_dim_values(x, dim=dim, keepdim=False, values=values, indices=indices)
    ref_out = torch.nanmedian(x, dim=dim, keepdim=False)
    ref_out = (ref_out.values, ref_out.indices)

    torch.testing.assert_close(values, ref_out[0])
    torch.testing.assert_close(indices, ref_out[1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_nanmedian_infinity_sentinel(dtype):
    x = torch.tensor([1.0, float("inf"), float("inf")], dtype=dtype)
    torch.testing.assert_close(nanmedian(x), torch.nanmedian(x))
    x = torch.stack(
        (x, torch.tensor([float("nan"), float("inf"), float("inf")], dtype=dtype))
    )
    result = nanmedian_dim(x, dim=1)
    expected = torch.nanmedian(x, dim=1)
    torch.testing.assert_close(result.values, expected.values)
    # Duplicate values can have different valid indices; require an active lane.
    assert torch.all((result.indices >= 0) & (result.indices < x.shape[1]))
    torch.testing.assert_close(
        x.gather(1, result.indices[:, None])[:, 0], expected.values
    )


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize(
    "strided_values,strided_indices", [(True, False), (False, True), (True, True)]
)
def test_nanmedian_noncontiguous_out(keepdim, strided_values, strided_indices):
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    values = (
        torch.full((3, 2), -100.0).t() if strided_values else torch.full((2, 3), -100.0)
    )
    indices = (
        torch.full((3, 2), -1, dtype=torch.int64).t()
        if strided_indices
        else torch.full((2, 3), -1, dtype=torch.int64)
    )
    if keepdim:
        values = values.unsqueeze(-1)
        indices = indices.unsqueeze(-1)
    result = nanmedian_dim_values(
        x, dim=2, keepdim=keepdim, values=values, indices=indices
    )
    expected = torch.nanmedian(x, dim=2, keepdim=keepdim)
    assert result.values is values
    assert result.indices is indices
    torch.testing.assert_close(values, expected.values)
    torch.testing.assert_close(indices, expected.indices)
