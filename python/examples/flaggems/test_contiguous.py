import torch

from .contiguous import contiguous


def test_contiguous_already_contiguous():
    x = torch.randn(4, 8, device="cpu", dtype=torch.float32)
    assert x.is_contiguous()
    out = contiguous(x)
    assert out is x


def test_contiguous_transpose():
    x = torch.randn(4, 8, device="cpu", dtype=torch.float32)
    x_t = x.t()
    assert not x_t.is_contiguous()
    out = contiguous(x_t)
    assert out.is_contiguous()
    torch.testing.assert_close(out, x_t)


def test_contiguous_broadcast_shape():
    x = torch.randn(1, 8, device="cpu", dtype=torch.float32)
    x_expanded = x.expand(4, 8)
    assert not x_expanded.is_contiguous()
    out = contiguous(x_expanded)
    assert out.is_contiguous()
    torch.testing.assert_close(out, x_expanded)
