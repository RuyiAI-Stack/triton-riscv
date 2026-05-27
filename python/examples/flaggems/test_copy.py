import torch

from .copy import copy, copy_


def test_copy_simple():
    x = torch.randn(16, 256, dtype=torch.float32, device="cpu")
    out = torch.empty_like(x)
    copy_(out, x)
    torch.testing.assert_close(out, x)


def test_copy_functional():
    template = torch.empty(4, 16, dtype=torch.float32, device="cpu")
    src = torch.randn(4, 16, device="cpu")
    out = copy(template, src)
    torch.testing.assert_close(out, src)


def test_copy_broadcast():
    dst = torch.empty(4, 16, dtype=torch.float32, device="cpu")
    src = torch.randn(16, device="cpu")
    copy_(dst, src)
    assert dst.shape == (4, 16)
