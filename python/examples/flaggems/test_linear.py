import torch
import torch.nn.functional as F

from .linear import linear


def test_linear():
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float32, device="cpu")
    weight = torch.randn(3, 8, dtype=torch.float32, device="cpu")
    bias = torch.randn(3, dtype=torch.float32, device="cpu")

    out = linear(x, weight, bias)
    ref = F.linear(x, weight, bias)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_linear_batched_noncontiguous_input():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 8, dtype=torch.float32, device="cpu").transpose(0, 1)
    weight = torch.randn(3, 8, dtype=torch.float32, device="cpu")
    bias = torch.randn(3, dtype=torch.float32, device="cpu")

    out = linear(x, weight, bias)
    ref = F.linear(x, weight, bias)

    assert not x.is_contiguous()
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
