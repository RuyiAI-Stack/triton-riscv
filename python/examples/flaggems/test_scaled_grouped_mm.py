import torch

from .scaled_grouped_mm import scaled_grouped_mm


def test_scaled_grouped_mm_single_group():
    torch.manual_seed(0)
    a = torch.randn(1, 2, 4, dtype=torch.float32, device="cpu")
    b = torch.randn(1, 4, 3, dtype=torch.float32, device="cpu")
    scale_a = torch.ones(1, 2, dtype=torch.float32, device="cpu")
    scale_b = torch.ones(1, 3, dtype=torch.float32, device="cpu")
    bias = torch.randn(3, dtype=torch.float32, device="cpu")

    out = scaled_grouped_mm(a, b, scale_a, scale_b, bias=bias)
    ref = torch.matmul(a, b) + bias

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
