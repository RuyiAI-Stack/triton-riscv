import importlib

import pytest
import torch

from ._fake_quantize_learnable_per_channel_affine import (
    _fake_quantize_learnable_per_channel_affine,
)


def _ref(x, scale, zp, axis, qmin, qmax):
    return torch.ops.aten._fake_quantize_learnable_per_channel_affine(
        x, scale, zp, axis, qmin, qmax
    )


def test_learnable_per_channel_matches_aten_axis0():
    torch.manual_seed(0)
    x = torch.randn(3, 17)
    scale = torch.tensor([0.1, 0.2, 0.3])
    zp = torch.tensor([0.0, 1.0, 2.0])
    out = _fake_quantize_learnable_per_channel_affine(x, scale, zp, 0, -128, 127)
    ref = _ref(x, scale, zp, 0, -128, 127)
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_learnable_per_channel_axis_variations():
    torch.manual_seed(1)
    x3d = torch.randn(2, 3, 4)
    for axis, n in ((0, 2), (1, 3), (2, 4)):
        scale = torch.full((n,), 0.1)
        zp = torch.arange(n, dtype=torch.float32)
        out = _fake_quantize_learnable_per_channel_affine(
            x3d, scale, zp, axis, -128, 127
        )
        ref = _ref(x3d, scale, zp, axis, -128, 127)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"axis={axis}")


def test_learnable_per_channel_negative_axis_normalized():
    # the wrapper normalizes axis = -1 to the trailing dimension
    torch.manual_seed(2)
    x = torch.randn(3, 17)
    scale = torch.full((17,), 0.1)
    zp = torch.zeros(17)
    out = _fake_quantize_learnable_per_channel_affine(x, scale, zp, -1, -128, 127)
    ref = _ref(x, scale, zp, x.dim() - 1, -128, 127)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_learnable_per_channel_zero_point_rounding_and_clamping():
    torch.manual_seed(3)
    x = torch.randn(3, 5)
    scale = torch.tensor([0.1, 0.2, 0.3])
    for zp in (
        torch.tensor([0.4, 1.6, -0.6]),
        torch.tensor([2.5, 2.4, -2.5]),
        torch.tensor([300.0, -300.0, 0.0]),
        torch.tensor([-128.0, 127.0, 0.0]),
    ):
        out = _fake_quantize_learnable_per_channel_affine(x, scale, zp, 0, -128, 127)
        ref = _ref(x, scale, zp, 0, -128, 127)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_learnable_per_channel_saturation_and_uint8_range():
    x = torch.tensor([[1e4, -1e4, 0.25, 0.26, 0.05]])
    scale = torch.tensor([0.1])
    zp = torch.tensor([0.0])
    out = _fake_quantize_learnable_per_channel_affine(x, scale, zp, 0, 0, 255)
    ref = _ref(x, scale, zp, 0, 0, 255)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(out[0, :2], torch.tensor([25.5, 0.0]), rtol=0, atol=0)


def test_learnable_per_channel_quant_min_equals_quant_max():
    x = torch.tensor([[-5.0, 0.0, 5.0]])
    scale = torch.tensor([0.1])
    zp = torch.tensor([5.0])
    out = _fake_quantize_learnable_per_channel_affine(x, scale, zp, 0, 5, 5)
    ref = _ref(x, scale, zp, 0, 5, 5)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(out, torch.zeros_like(x), rtol=0, atol=0)


def test_learnable_per_channel_extreme_scales():
    torch.manual_seed(4)
    x = torch.randn(3, 9)
    for scale in (torch.full((3,), 1e-4), torch.full((3,), 32.0)):
        zp = torch.zeros(3)
        out = _fake_quantize_learnable_per_channel_affine(x, scale, zp, 0, -128, 127)
        ref = _ref(x, scale, zp, 0, -128, 127)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_learnable_per_channel_float16():
    torch.manual_seed(5)
    x = torch.randn(3, 17, dtype=torch.float16)
    scale = torch.tensor([0.1, 0.2, 0.3])
    zp = torch.tensor([0.0, 1.0, 2.0])
    out = _fake_quantize_learnable_per_channel_affine(x, scale, zp, 0, -128, 127)
    ref = _ref(x, scale, zp, 0, -128, 127)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_learnable_per_channel_grad_factor_and_empty():
    torch.manual_seed(6)
    x = torch.randn(3, 5)
    scale = torch.tensor([0.1, 0.2, 0.3])
    zp = torch.tensor([0.0, 1.0, 2.0])
    a = _fake_quantize_learnable_per_channel_affine(x, scale, zp, 0, -128, 127, 0.5)
    b = _fake_quantize_learnable_per_channel_affine(x, scale, zp, 0, -128, 127, 2.0)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    out = _fake_quantize_learnable_per_channel_affine(
        torch.empty(0, 3), scale, zp, 1, -128, 127
    )
    assert out.shape == (0, 3)


def test_learnable_per_channel_rejects_non_float_input():
    with pytest.raises(TypeError, match="only floating-point dtypes are supported"):
        _fake_quantize_learnable_per_channel_affine(
            torch.zeros(3, 5, dtype=torch.int32),
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([0.0, 1.0, 2.0]),
            0,
            -128,
            127,
        )


def test_learnable_per_channel_triton_path(monkeypatch):
    mod = importlib.import_module(
        "flaggems._fake_quantize_learnable_per_channel_affine"
    )
    launches = []
    kernel = mod.fake_quantize_learnable_per_channel_affine_forward_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    x = torch.randn(3, 17)
    scale = torch.tensor([0.1, 0.2, 0.3])
    zp = torch.tensor([0.0, 1.0, 2.0])
    out = _fake_quantize_learnable_per_channel_affine(x, scale, zp, 0, -128, 127)
    assert len(launches) >= 1
    torch.testing.assert_close(out, _ref(x, scale, zp, 0, -128, 127), rtol=0, atol=0)
