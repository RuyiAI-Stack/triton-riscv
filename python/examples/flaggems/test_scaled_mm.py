import torch

from .scaled_mm import scaled_mm, scaled_mm_out


def test_scaled_mm_unit_scales():
    torch.manual_seed(0)
    a = torch.randn(3, 4, dtype=torch.float32, device="cpu")
    b = torch.randn(4, 5, dtype=torch.float32, device="cpu")
    scale_a = torch.ones(3, dtype=torch.float32, device="cpu")
    scale_b = torch.ones(5, dtype=torch.float32, device="cpu")
    bias = torch.randn(5, dtype=torch.float32, device="cpu")

    out = scaled_mm(a, b, scale_a, scale_b, bias=bias)
    ref = a @ b + bias

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_scaled_mm_out():
    a = torch.eye(3, dtype=torch.float32, device="cpu")
    b = torch.ones(3, 2, dtype=torch.float32, device="cpu")
    out = torch.empty(3, 2, dtype=torch.float32, device="cpu")

    ret = scaled_mm_out(a, b, torch.ones(3), torch.ones(2), out=out)

    assert ret is out
    torch.testing.assert_close(out, a @ b)
