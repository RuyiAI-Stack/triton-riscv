import pytest
import torch

from .pow import (
    pow,
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_pow_tt(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu").abs()
    exponent = torch.randn(shape, dtype=torch.float32, device="cpu").abs()

    ref_out = torch.pow(x, exponent)
    tri_out = pow(x, exponent)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_pow_ts(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu").abs()
    exponent = 3.0

    ref_out = torch.pow(x, exponent)
    tri_out = pow(x, exponent)

    # Triton pow uses exp(exponent * log(x)) which may have small diffs
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_pow_st(shape):
    torch.manual_seed(0)
    x = 2.0
    exponent = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.pow(x, exponent)
    tri_out = pow(x, exponent)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "x_shape, exp_shape", [((256,), (1,)), ((1,), (512,))]
)
def test_pow_broadcast(x_shape, exp_shape):
    torch.manual_seed(0)
    x = torch.randn(x_shape, dtype=torch.float32, device="cpu").abs()
    exponent = torch.randn(exp_shape, dtype=torch.float32, device="cpu").abs()

    ref_out = torch.pow(x, exponent)
    tri_out = pow(x, exponent)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(512,), (1024,)])
def test_pow_tensor_tensor_inplace(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu").abs()
    exponent = torch.randn(shape, dtype=torch.float32, device="cpu").abs()

    ref = x.clone()
    ref.copy_(torch.pow(ref, exponent))
    tri = x.clone()
    result = pow_tensor_tensor_(tri, exponent)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)
    assert result is tri, "pow_tensor_tensor_ must return the input tensor"


@pytest.mark.parametrize("shape", [(512,), (1024,)])
def test_pow_tensor_scalar_inplace(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu").abs()
    exponent = 3.0

    ref = x.clone()
    ref.copy_(torch.pow(ref, exponent))
    tri = x.clone()
    result = pow_tensor_scalar_(tri, exponent)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)
    assert result is tri, "pow_tensor_scalar_ must return the input tensor"


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_pow_scalar_explicit(shape):
    """Test pow_scalar (scalar base, tensor exponent) by direct function name."""
    torch.manual_seed(0)
    base = 2.0
    exponent = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.pow(base, exponent)
    tri_out = pow_scalar(base, exponent)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_pow_tensor_scalar_explicit(shape):
    """Test pow_tensor_scalar (tensor base, scalar exponent) by direct function name."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu").abs()
    exponent = 3.0

    ref_out = torch.pow(x, exponent)
    tri_out = pow_tensor_scalar(x, exponent)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_pow_tensor_tensor_explicit(shape):
    """Test pow_tensor_tensor (tensor base, tensor exponent) by direct function name."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu").abs()
    exponent = torch.randn(shape, dtype=torch.float32, device="cpu").abs()

    ref_out = torch.pow(x, exponent)
    tri_out = pow_tensor_tensor(x, exponent)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)
