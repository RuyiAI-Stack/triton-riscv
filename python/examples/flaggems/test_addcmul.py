import pytest
import torch

from .addcmul import addcmul, addcmul_out


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
@pytest.mark.parametrize("value", [1.0, 2.5])
def test_addcmul(shape, value):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    t1 = torch.randn(shape, dtype=torch.float32, device="cpu")
    t2 = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.addcmul(x, t1, t2, value=value)
    tri_out = addcmul(x, t1, t2, value=value)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256)])
@pytest.mark.parametrize("value", [2.5])
def test_addcmul_out(shape, value):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    t1 = torch.randn(shape, dtype=torch.float32, device="cpu")
    t2 = torch.randn(shape, dtype=torch.float32, device="cpu")
    out = torch.empty_like(x)

    ref_out = torch.addcmul(x, t1, t2, value=value)
    addcmul_out(x, t1, t2, value=value, out=out)

    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
