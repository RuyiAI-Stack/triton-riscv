import pytest
import torch

from .softshrink import softshrink, softshrink_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_softshrink(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    lambd = 0.5
    ref_out = torch.nn.functional.softshrink(x, lambd=lambd)
    tri_out = softshrink(x, lambd=lambd)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_softshrink_lambd(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    lambd = 1.0
    ref_out = torch.nn.functional.softshrink(x, lambd=lambd)
    tri_out = softshrink(x, lambd=lambd)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_softshrink_out(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    lambd = 0.5
    out = torch.empty_like(x)
    ref_out = torch.nn.functional.softshrink(x, lambd=lambd)
    softshrink_out(x, lambd=lambd, out=out)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_softshrink_fp16(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float16)
    lambd = 0.5
    ref_out = torch.nn.functional.softshrink(x, lambd=lambd)
    tri_out = softshrink(x, lambd=lambd)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)
