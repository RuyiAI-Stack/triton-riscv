import pytest
import torch

from .clamp import (
    clamp,
    clamp_,
    clamp_min,
    clamp_min_,
    clamp_tensor,
    clamp_tensor_,
)


@pytest.mark.parametrize("shape", [(16, 256), (1024,)])
def test_clamp_both(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref = torch.clamp(x, -0.5, 0.5)
    tri = clamp(x, -0.5, 0.5)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (1024,)])
def test_clamp_min_only(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref = torch.clamp(x, min=0.0)
    tri = clamp(x, mini=0.0)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256)])
def test_clamp_max_only(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref = torch.clamp(x, max=0.0)
    tri = clamp(x, maxi=0.0)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_clamp_inplace():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    x_ref.clamp_(-0.3, 0.3)
    clamp_(x, -0.3, 0.3)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256)])
def test_clamp_tensor_both(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    mini = torch.full(shape, -0.5, device="cpu")
    maxi = torch.full(shape, 0.5, device="cpu")
    ref = torch.clamp(x, -0.5, 0.5)
    tri = clamp_tensor(x, mini, maxi)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256)])
def test_clamp_tensor_min_only(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    mini = torch.full(shape, 0.0, device="cpu")
    ref = torch.clamp(x, min=0.0)
    tri = clamp_tensor(x, mini=mini)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256)])
def test_clamp_tensor_max_only(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    maxi = torch.full(shape, 0.0, device="cpu")
    ref = torch.clamp(x, max=0.0)
    tri = clamp_tensor(x, maxi=maxi)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_clamp_tensor_inplace():
    torch.manual_seed(0)
    x = torch.randn(16, 256, dtype=torch.float32, device="cpu")
    mini = torch.full((16, 256), -0.3, device="cpu")
    maxi = torch.full((16, 256), 0.3, device="cpu")
    x_ref = x.clone()
    x_ref.clamp_(-0.3, 0.3)
    clamp_tensor_(x, mini, maxi)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (1024,)])
def test_clamp_min_scalar(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    min_val = 0.0
    ref = torch.clamp(x, min=min_val)
    tri = clamp_min(x, min_val)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_clamp_min_scalar_inplace():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    x_ref.clamp_(min=-0.3)
    clamp_min_(x, -0.3)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
