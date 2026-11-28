import importlib

import pytest
import torch

from ._fake_quantize_learnable_per_channel_affine_backward import (
    _fake_quantize_learnable_per_channel_affine_backward,
)


def _ref(grad, x, scale, zp, axis, qmin, qmax, grad_factor=1.0):
    return torch.ops.aten._fake_quantize_learnable_per_channel_affine_backward(
        grad, x, scale, zp, axis, qmin, qmax, grad_factor
    )


# dScale/dZp are fp32 reductions over the non-axis dimensions: the kernel's
# summation order differs from aten's, so last-ulp differences (measured
# <=2e-7) are expected; dX stays bit-exact (elementwise).


def test_learnable_per_channel_backward_matches_aten_axis0():
    torch.manual_seed(0)
    grad = torch.randn(3, 17)
    x = torch.randn(3, 17)
    scale = torch.tensor([0.1, 0.2, 0.3])
    zp = torch.tensor([0.0, 1.0, 2.0])
    dX, dScale, dZp = _fake_quantize_learnable_per_channel_affine_backward(
        grad, x, scale, zp, 0, -128, 127
    )
    rX, rScale, rZp = _ref(grad, x, scale, zp, 0, -128, 127)
    assert dX.shape == x.shape
    assert dScale.shape == (3,) and dZp.shape == (3,)
    torch.testing.assert_close(dX, rX, rtol=0, atol=0)
    torch.testing.assert_close(dScale, rScale, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(dZp, rZp, rtol=1e-5, atol=1e-6)


def test_learnable_per_channel_backward_axis_variations():
    torch.manual_seed(1)
    x3d = torch.randn(2, 3, 4)
    grad3d = torch.randn(2, 3, 4)
    for axis, n in ((0, 2), (1, 3), (2, 4)):
        scale = torch.full((n,), 0.1)
        zp = torch.arange(n, dtype=torch.float32)
        dX, dScale, dZp = _fake_quantize_learnable_per_channel_affine_backward(
            grad3d, x3d, scale, zp, axis, -128, 127
        )
        rX, rScale, rZp = _ref(grad3d, x3d, scale, zp, axis, -128, 127)
        assert dScale.shape == (n,) and dZp.shape == (n,)
        torch.testing.assert_close(dX, rX, rtol=0, atol=0, msg=f"axis={axis}")
        torch.testing.assert_close(dScale, rScale, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(dZp, rZp, rtol=1e-5, atol=1e-6)


def test_learnable_per_channel_backward_saturation_branches():
    # saturated channels: dX zero, dScale/dZp pick up the clamped-value branch
    # (per-channel axis=0 on a (2, 2) input needs one qparam per channel)
    grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x = torch.tensor([[1e4, -1e4], [0.25, 0.26]])
    scale = torch.tensor([0.1, 0.1])
    zp = torch.tensor([0.0, 0.0])
    dX, dScale, dZp = _fake_quantize_learnable_per_channel_affine_backward(
        grad, x, scale, zp, 0, -128, 127
    )
    rX, rScale, rZp = _ref(grad, x, scale, zp, 0, -128, 127)
    torch.testing.assert_close(dX, rX, rtol=0, atol=0)
    assert dX[0, 0].item() == 0.0 and dX[0, 1].item() == 0.0
    assert dX[1, 0].item() != 0.0
    torch.testing.assert_close(dScale, rScale, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(dZp, rZp, rtol=1e-5, atol=1e-6)


def test_learnable_per_channel_backward_zero_point_rounding():
    torch.manual_seed(2)
    grad = torch.randn(3, 5)
    x = torch.randn(3, 5)
    scale = torch.tensor([0.1, 0.2, 0.3])
    for zp in (torch.tensor([0.4, 1.6, -0.6]), torch.tensor([300.0, -300.0, 0.0])):
        dX, dScale, dZp = _fake_quantize_learnable_per_channel_affine_backward(
            grad, x, scale, zp, 0, -128, 127
        )
        rX, rScale, rZp = _ref(grad, x, scale, zp, 0, -128, 127)
        torch.testing.assert_close(dX, rX, rtol=0, atol=0)
        torch.testing.assert_close(dScale, rScale, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(dZp, rZp, rtol=1e-5, atol=1e-6)


def test_learnable_per_channel_backward_grad_factor():
    torch.manual_seed(3)
    grad = torch.randn(3, 17)
    x = torch.randn(3, 17)
    scale = torch.tensor([0.1, 0.2, 0.3])
    zp = torch.tensor([0.0, 1.0, 2.0])
    aX, aS, aZ = _fake_quantize_learnable_per_channel_affine_backward(
        grad, x, scale, zp, 0, -128, 127, 1.0
    )
    bX, bS, bZ = _fake_quantize_learnable_per_channel_affine_backward(
        grad, x, scale, zp, 0, -128, 127, 0.5
    )
    torch.testing.assert_close(bX, aX, rtol=0, atol=0)
    torch.testing.assert_close(bS, aS * 0.5, rtol=0, atol=0)
    torch.testing.assert_close(bZ, aZ * 0.5, rtol=0, atol=0)
    rX, rS, rZ = _ref(grad, x, scale, zp, 0, -128, 127, 0.5)
    torch.testing.assert_close(bS, rS, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(bZ, rZ, rtol=1e-5, atol=1e-6)


def test_learnable_per_channel_backward_empty():
    dX, dScale, dZp = _fake_quantize_learnable_per_channel_affine_backward(
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.tensor([0.1, 0.2, 0.3]),
        torch.tensor([0.0, 1.0, 2.0]),
        1,
        -128,
        127,
    )
    assert dX.shape == (0, 3)
    assert dScale.shape == (3,) and dZp.shape == (3,)
    torch.testing.assert_close(dScale, torch.zeros(3), rtol=0, atol=0)
    torch.testing.assert_close(dZp, torch.zeros(3), rtol=0, atol=0)


def test_learnable_per_channel_backward_requires_float32():
    grad = torch.randn(3, 5)
    x = torch.randn(3, 5)
    scale = torch.tensor([0.1, 0.2, 0.3])
    zp = torch.tensor([0.0, 1.0, 2.0])
    with pytest.raises(AssertionError, match="grad must be float32"):
        _fake_quantize_learnable_per_channel_affine_backward(
            grad.double(), x, scale, zp, 0, -128, 127
        )
    with pytest.raises(AssertionError, match="self must be float32"):
        _fake_quantize_learnable_per_channel_affine_backward(
            grad, x.double(), scale, zp, 0, -128, 127
        )


def test_learnable_per_channel_backward_triton_path(monkeypatch):
    mod = importlib.import_module(
        "flaggems._fake_quantize_learnable_per_channel_affine_backward"
    )
    launches = []
    kernel = mod._fake_quantize_learnable_per_channel_affine_backward_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    grad = torch.randn(3, 17)
    x = torch.randn(3, 17)
    scale = torch.tensor([0.1, 0.2, 0.3])
    zp = torch.tensor([0.0, 1.0, 2.0])
    dX, dScale, dZp = _fake_quantize_learnable_per_channel_affine_backward(
        grad, x, scale, zp, 0, -128, 127
    )
    assert len(launches) >= 1
    rX, rScale, rZp = _ref(grad, x, scale, zp, 0, -128, 127)
    torch.testing.assert_close(dX, rX, rtol=0, atol=0)
    torch.testing.assert_close(dScale, rScale, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(dZp, rZp, rtol=1e-5, atol=1e-6)
