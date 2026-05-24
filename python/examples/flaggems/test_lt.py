import pytest
import torch

from .lt import lt, lt_scalar


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_lt(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = x < y
    tri = lt(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_lt_scalar():
    x = torch.tensor([0.0, 1.0, 2.0, 0.0], dtype=torch.float32, device="cpu")
    tri = lt_scalar(x, 1.0)
    ref = x < 1.0
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
