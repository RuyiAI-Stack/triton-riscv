import pytest
import torch

from .clip import clip, clip_


@pytest.mark.parametrize("shape", [(16, 256)])
def test_clip(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref = torch.clip(x, -0.5, 0.5)
    tri = clip(x, -0.5, 0.5)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (64, 128)])
def test_clip_inplace_both(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    x_clone = x.clone()
    ref = torch.clip(x_clone, -0.5, 0.5)
    result = clip_(x, -0.5, 0.5)
    torch.testing.assert_close(result, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256)])
def test_clip_inplace_min_only(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    x_clone = x.clone()
    ref = torch.clip(x_clone, min=-0.3)
    result = clip_(x, mini=-0.3)
    torch.testing.assert_close(result, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256)])
def test_clip_inplace_max_only(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    x_clone = x.clone()
    ref = torch.clip(x_clone, max=0.3)
    result = clip_(x, maxi=0.3)
    torch.testing.assert_close(result, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x, ref, rtol=1e-4, atol=1e-4)
