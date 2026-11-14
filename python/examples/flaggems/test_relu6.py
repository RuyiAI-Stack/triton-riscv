import pytest
import torch
import torch.nn.functional as F

from .relu6 import relu6


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_relu6(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref_out = F.relu6(x)
    tri_out = relu6(x)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256)])
def test_relu6_2d(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref_out = F.relu6(x)
    tri_out = relu6(x)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_relu6_nan():
    x = torch.tensor([float("nan"), float("inf"), float("-inf"), 3.0])
    torch.testing.assert_close(relu6(x), torch.nn.functional.relu6(x), equal_nan=True)
