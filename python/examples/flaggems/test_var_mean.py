import pytest
import torch

from .var_mean import var_mean


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("correction", [0.0, 1.0])
def test_var_mean_1d(size, correction):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_var, ref_mean = torch.var_mean(x, correction=correction)
    tri_var, tri_mean = var_mean(x, correction=correction)

    torch.testing.assert_close(tri_mean, ref_mean, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_var, ref_var, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_var_mean_1d_keepdim(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_var, ref_mean = torch.var_mean(x, keepdim=True)
    tri_var, tri_mean = var_mean(x, keepdim=True)

    assert tri_mean.shape == ref_mean.shape
    assert tri_var.shape == ref_var.shape
    torch.testing.assert_close(tri_mean, ref_mean, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_var, ref_var, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(32, 16), (64, 64)])
def test_var_mean_2d(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_var, ref_mean = torch.var_mean(x)
    tri_var, tri_mean = var_mean(x)

    torch.testing.assert_close(tri_mean, ref_mean, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_var, ref_var, rtol=1e-4, atol=1e-4)


def test_var_mean_large_constant_is_numerically_stable():
    x = torch.full((4096,), 1.0e8, dtype=torch.float32)

    tri_var, tri_mean = var_mean(x, correction=0)
    ref_var, ref_mean = torch.var_mean(x, correction=0)

    torch.testing.assert_close(tri_mean, ref_mean)
    torch.testing.assert_close(tri_var, ref_var)
    assert tri_var.item() == 0.0
