import importlib

import pytest
import torch

from ._fake_quantize_learnable_per_tensor_affine_backward import (
    _fake_quantize_learnable_per_tensor_affine_backward,
)


def _ref(grad, x, scale, zp, qmin, qmax, grad_factor=1.0):
    return torch.ops.aten._fake_quantize_learnable_per_tensor_affine_backward(
        grad, x, torch.tensor([scale]), torch.tensor([zp]), qmin, qmax, grad_factor
    )


# dS/dZ are single-element fp32 reductions over the whole tensor: the kernel's
# tile reduction order differs from aten's, so last-ulp differences (measured
# <=1e-6 abs on scalars) are expected; dX comparisons stay bit-exact.


def test_learnable_per_tensor_backward_matches_aten():
    torch.manual_seed(0)
    for shape in ((7,), (3, 17), (2, 3, 5)):
        grad = torch.randn(shape)
        x = torch.randn(shape)
        for scale, zp, qmin, qmax in (
            (0.1, 0.0, -128, 127),
            (0.05, 128.0, 0, 255),
            (0.03125, -3.0, -128, 127),
        ):
            dX, dS, dZ = _fake_quantize_learnable_per_tensor_affine_backward(
                grad, x, scale, zp, qmin, qmax
            )
            rX, rS, rZ = _ref(grad, x, scale, zp, qmin, qmax)
            assert dX.dtype == torch.float32
            assert dS.dtype == torch.float32 and dZ.dtype == torch.float32
            assert dS.shape == (1,) and dZ.shape == (1,)
            torch.testing.assert_close(dX, rX, rtol=0, atol=0)
            torch.testing.assert_close(dS, rS, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(dZ, rZ, rtol=1e-5, atol=1e-6)


def test_learnable_per_tensor_backward_grad_zero_where_saturated():
    # far-outside-range inputs contribute zero input-gradient but nonzero
    # scale/zero-point gradients
    grad = torch.tensor([1.0, 2.0, 3.0])
    x = torch.tensor([1e4, -1e4, 0.25])
    dX, dS, dZ = _fake_quantize_learnable_per_tensor_affine_backward(
        grad, x, 0.1, 0.0, -128, 127
    )
    rX, rS, rZ = _ref(grad, x, 0.1, 0.0, -128, 127)
    torch.testing.assert_close(dX, rX, rtol=0, atol=0)
    assert dX[0].item() == 0.0 and dX[1].item() == 0.0
    assert dX[2].item() != 0.0
    torch.testing.assert_close(dS, rS, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(dZ, rZ, rtol=1e-5, atol=1e-6)


def test_learnable_per_tensor_backward_zero_point_rounding_and_clamping():
    torch.manual_seed(1)
    grad = torch.randn(2, 9)
    x = torch.randn(2, 9)
    for zp, qmin, qmax in ((2.5, -128, 127), (300.0, 0, 255), (-300.0, -128, 127)):
        dX, dS, dZ = _fake_quantize_learnable_per_tensor_affine_backward(
            grad, x, 0.1, zp, qmin, qmax
        )
        rX, rS, rZ = _ref(grad, x, 0.1, zp, qmin, qmax)
        torch.testing.assert_close(dX, rX, rtol=0, atol=0)
        torch.testing.assert_close(dS, rS, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(dZ, rZ, rtol=1e-5, atol=1e-6)


def test_learnable_per_tensor_backward_grad_factor():
    torch.manual_seed(2)
    grad = torch.randn(3, 17)
    x = torch.randn(3, 17)
    aX, aS, aZ = _fake_quantize_learnable_per_tensor_affine_backward(
        grad, x, 0.1, 0.0, -128, 127, 1.0
    )
    bX, bS, bZ = _fake_quantize_learnable_per_tensor_affine_backward(
        grad, x, 0.1, 0.0, -128, 127, 0.5
    )
    # grad_factor scales dS/dZ but leaves dX untouched
    torch.testing.assert_close(bX, aX, rtol=0, atol=0)
    torch.testing.assert_close(bS, aS * 0.5, rtol=0, atol=0)
    torch.testing.assert_close(bZ, aZ * 0.5, rtol=0, atol=0)
    rX, rS, rZ = _ref(grad, x, 0.1, 0.0, -128, 127, 0.5)
    torch.testing.assert_close(bS, rS, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(bZ, rZ, rtol=1e-5, atol=1e-6)


def test_learnable_per_tensor_backward_bfloat16_inputs():
    torch.manual_seed(3)
    grad = torch.randn(2, 17, dtype=torch.bfloat16)
    x = torch.randn(2, 17, dtype=torch.bfloat16)
    dX, dS, dZ = _fake_quantize_learnable_per_tensor_affine_backward(
        grad, x, 0.1, 0.0, -128, 127
    )
    rX, rS, rZ = _ref(grad, x, 0.1, 0.0, -128, 127)
    # gradients are promoted to float32
    assert dX.dtype == torch.float32
    torch.testing.assert_close(dX, rX, rtol=0, atol=0)
    torch.testing.assert_close(dS, rS, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(dZ, rZ, rtol=1e-5, atol=1e-6)


def test_learnable_per_tensor_backward_empty():
    # empty input: all three gradients are defined and zero for the parameters
    dX, dS, dZ = _fake_quantize_learnable_per_tensor_affine_backward(
        torch.empty(0), torch.empty(0), 0.1, 0.0, -128, 127
    )
    assert dX.shape == (0,)
    assert dS.shape == (1,) and dZ.shape == (1,)
    assert dS.item() == 0.0 and dZ.item() == 0.0


def test_learnable_per_tensor_backward_errors():
    grad = torch.randn(3)
    x = torch.randn(3)
    with pytest.raises(ValueError, match="All inputs must be floating-point tensors"):
        _fake_quantize_learnable_per_tensor_affine_backward(
            grad, torch.arange(3), 0.1, 0.0, -128, 127
        )
    with pytest.raises(ValueError, match="expected float32 or bfloat16"):
        _fake_quantize_learnable_per_tensor_affine_backward(
            grad, x.half(), 0.1, 0.0, -128, 127
        )


def test_learnable_per_tensor_backward_triton_path(monkeypatch):
    mod = importlib.import_module(
        "flaggems._fake_quantize_learnable_per_tensor_affine_backward"
    )
    # small tensors take the fused _fq_tiny_kernel; larger ones take the
    # main kernel plus the _reduce_kernel pass
    for kernel_name in (
        "_fq_tiny_kernel",
        "_fake_quantize_learnable_per_tensor_affine_backward_kernel",
        "_reduce_kernel",
    ):
        kernel = getattr(mod, kernel_name)
        assert hasattr(kernel, "run"), f"{kernel_name} must be a triton kernel"

    launches = []
    for kernel_name in (
        "_fq_tiny_kernel",
        "_fake_quantize_learnable_per_tensor_affine_backward_kernel",
        "_reduce_kernel",
    ):
        kernel = getattr(mod, kernel_name)
        real_run = kernel.run

        def spy_run(*args, _real=real_run, **kwargs):
            launches.append(kwargs.get("grid"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
    grad = torch.randn(3, 17)
    x = torch.randn(3, 17)
    dX, dS, dZ = _fake_quantize_learnable_per_tensor_affine_backward(
        grad, x, 0.1, 0.0, -128, 127
    )
    assert len(launches) >= 1, "a backward kernel must be launched"
    rX, rS, rZ = _ref(grad, x, 0.1, 0.0, -128, 127)
    torch.testing.assert_close(dX, rX, rtol=0, atol=0)
    torch.testing.assert_close(dS, rS, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(dZ, rZ, rtol=1e-5, atol=1e-6)
