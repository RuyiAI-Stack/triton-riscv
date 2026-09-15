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
