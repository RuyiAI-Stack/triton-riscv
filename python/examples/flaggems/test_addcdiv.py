import pytest
import torch

from .addcdiv import addcdiv, addcdiv_out


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
@pytest.mark.parametrize("value", [1.0, 2.5])
def test_addcdiv(shape, value):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    t1 = torch.randn(shape, dtype=torch.float32, device="cpu")
    t2 = torch.randn(shape, dtype=torch.float32, device="cpu")
    # avoid division by zero
    t2 = torch.where(torch.abs(t2) < 0.1, torch.sign(t2) * 0.1 + 0.1, t2)

    ref_out = torch.addcdiv(x, t1, t2, value=value)
    tri_out = addcdiv(x, t1, t2, value=value)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256)])
@pytest.mark.parametrize("value", [2.5])
def test_addcdiv_out(shape, value):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    t1 = torch.randn(shape, dtype=torch.float32, device="cpu")
    t2 = torch.randn(shape, dtype=torch.float32, device="cpu")
    t2 = torch.where(torch.abs(t2) < 0.1, torch.sign(t2) * 0.1 + 0.1, t2)
    out = torch.empty_like(x)

    ref_out = torch.addcdiv(x, t1, t2, value=value)
    addcdiv_out(x, t1, t2, value=value, out=out)

    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
