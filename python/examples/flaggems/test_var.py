import pytest
import torch

from .var import var, var_correction, var_dim


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_var(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.var(x)
    tri_out = var(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_var_correction_0(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.var(x, correction=0)
    tri_out = var(x, correction=0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_var_dim(size):
    torch.manual_seed(0)
    x = torch.randn(4, size, device="cpu", dtype=torch.float32)

    ref_out = torch.var(x, dim=1)
    tri_out = var(x, dim=[1])

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_var_correction_func(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.var(x, correction=2)
    tri_out = var_correction(x, correction=2)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_var_correction_func_0(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.var(x, correction=0)
    tri_out = var_correction(x, correction=0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_var_dim_func(size):
    torch.manual_seed(0)
    x = torch.randn(4, size, device="cpu", dtype=torch.float32)

    ref_out = torch.var(x, dim=1)
    tri_out = var_dim(x, dim=[1])

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_var_dim_func_keepdim(size):
    torch.manual_seed(0)
    x = torch.randn(4, size, device="cpu", dtype=torch.float32)

    ref_out = torch.var(x, dim=1, keepdim=True)
    tri_out = var_dim(x, dim=[1], keepdim=True)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_var_dim_func_multi_dim(size):
    torch.manual_seed(0)
    x = torch.randn(2, 3, size, device="cpu", dtype=torch.float32)

    ref_out = torch.var(x, dim=[0, 1])
    tri_out = var_dim(x, dim=[0, 1])

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_var_large_constant_is_numerically_stable():
    x = torch.full((4096,), 1.0e8, dtype=torch.float32)

    tri_out = var(x, correction=0)

    torch.testing.assert_close(tri_out, torch.var(x, correction=0))
    assert tri_out.item() == 0.0
