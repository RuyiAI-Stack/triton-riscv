import importlib

import torch

from ._fake_quantize_learnable_per_tensor_affine import (
    _fake_quantize_learnable_per_tensor_affine,
)


def _ref(x, scale, zp, qmin, qmax):
    return torch.ops.aten._fake_quantize_learnable_per_tensor_affine(
        x, torch.tensor([scale]), torch.tensor([zp]), qmin, qmax
    )


def test_learnable_per_tensor_matches_aten():
    torch.manual_seed(0)
    for shape in ((7,), (3, 17), (2, 3, 5)):
        x = torch.randn(shape)
        for scale, zp, qmin, qmax in (
            (0.1, 0.0, -128, 127),
            (0.05, 128.0, 0, 255),
            (0.03125, -3.0, -128, 127),
        ):
            out = _fake_quantize_learnable_per_tensor_affine(x, scale, zp, qmin, qmax)
            ref = _ref(x, scale, zp, qmin, qmax)
            assert out.dtype == x.dtype
            assert out.shape == x.shape
            torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_learnable_per_tensor_saturation_far_outside_range():
    x = torch.tensor([1e4, -1e4, 0.25, 0.26])
    out = _fake_quantize_learnable_per_tensor_affine(x, 0.1, 0.0, -128, 127)
    ref = _ref(x, 0.1, 0.0, -128, 127)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(out[:2], torch.tensor([12.7, -12.8]), rtol=0, atol=0)


def test_learnable_per_tensor_zero_point_rounding_and_clamping():
    torch.manual_seed(1)
    x = torch.randn(2, 9)
    # fractional zero point is rounded to nearest-even then clamped into range
    for zp, qmin, qmax in (
        (2.5, -128, 127),
        (2.4, -128, 127),
        (300.0, 0, 255),
        (-300.0, -128, 127),
        (-128.0, -128, 127),
        (127.0, -128, 127),
    ):
        out = _fake_quantize_learnable_per_tensor_affine(x, 0.1, zp, qmin, qmax)
        ref = _ref(x, 0.1, zp, qmin, qmax)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"zp={zp}")


def test_learnable_per_tensor_quant_min_equals_quant_max():
    x = torch.tensor([-5.0, 0.0, 5.0])
    out = _fake_quantize_learnable_per_tensor_affine(x, 0.1, 5.0, 5, 5)
    ref = _ref(x, 0.1, 5.0, 5, 5)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(out, torch.zeros(3), rtol=0, atol=0)


def test_learnable_per_tensor_extreme_scales():
    torch.manual_seed(2)
    x = torch.randn(2, 33)
    for scale in (1e-4, 16.0):
        out = _fake_quantize_learnable_per_tensor_affine(x, scale, 0.0, -128, 127)
        ref = _ref(x, scale, 0.0, -128, 127)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_learnable_per_tensor_tensor_valued_qparams():
    torch.manual_seed(3)
    x = torch.randn(5)
    out = _fake_quantize_learnable_per_tensor_affine(
        x, torch.tensor([0.1]), torch.tensor([2.0]), -128, 127
    )
    ref = _ref(x, 0.1, 2.0, -128, 127)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_learnable_per_tensor_grad_factor_does_not_affect_forward():
    torch.manual_seed(4)
    x = torch.randn(2, 17)
    a = _fake_quantize_learnable_per_tensor_affine(x, 0.1, 0.0, -128, 127, 0.5)
    b = _fake_quantize_learnable_per_tensor_affine(x, 0.1, 0.0, -128, 127, 2.0)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_learnable_per_tensor_float16_and_bfloat16():
    torch.manual_seed(5)
    for dtype in (torch.float16, torch.bfloat16):
        x = torch.randn(2, 17, dtype=dtype)
        out = _fake_quantize_learnable_per_tensor_affine(x, 0.1, 0.0, -128, 127)
        ref = _ref(x, 0.1, 0.0, -128, 127)
        assert out.dtype == dtype
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_learnable_per_tensor_empty():
    out = _fake_quantize_learnable_per_tensor_affine(
        torch.empty(0), 0.1, 0.0, -128, 127
    )
    assert out.shape == (0,)


def test_learnable_per_tensor_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._fake_quantize_learnable_per_tensor_affine")
    launches = []
    kernel = mod._fake_quantize_learnable_per_tensor_affine_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    x = torch.randn(3, 17)
    out = _fake_quantize_learnable_per_tensor_affine(x, 0.1, 0.0, -128, 127)
    assert len(launches) >= 1
    torch.testing.assert_close(out, _ref(x, 0.1, 0.0, -128, 127), rtol=0, atol=0)
