import importlib

import pytest
import torch

from .fake_quantize_per_tensor_affine import fake_quantize_per_tensor_affine


def test_fq_per_tensor_affine_matches_aten():
    torch.manual_seed(0)
    for shape in ((7,), (3, 17), (2, 3, 5)):
        x = torch.randn(shape)
        for scale, zp, qmin, qmax in (
            (0.1, 0, -128, 127),
            (0.05, 128, 0, 255),
            (0.03125, -3, -128, 127),
        ):
            out = fake_quantize_per_tensor_affine(x, scale, zp, qmin, qmax)
            ref = torch.fake_quantize_per_tensor_affine(x, scale, zp, qmin, qmax)
            torch.testing.assert_close(out, ref, rtol=0, atol=0)
            assert out.dtype == x.dtype
            assert out.shape == x.shape


def test_fq_per_tensor_affine_saturation_far_outside_range():
    x = torch.tensor([1e6, -1e6, 0.04, 0.05, 0.06])
    out = fake_quantize_per_tensor_affine(x, 0.1, 0, -128, 127)
    ref = torch.fake_quantize_per_tensor_affine(x, 0.1, 0, -128, 127)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    # saturated to the dequantized quant_max / quant_min (probed aten values)
    torch.testing.assert_close(out[:2], torch.tensor([12.7, -12.8]), rtol=0, atol=0)


def test_fq_per_tensor_affine_zero_point_at_clamp_edges():
    torch.manual_seed(1)
    x = torch.randn(2, 9)
    for zp, qmin, qmax in (
        (-128, -128, 127),
        (127, -128, 127),
        (0, 0, 255),
        (255, 0, 255),
    ):
        out = fake_quantize_per_tensor_affine(x, 0.1, zp, qmin, qmax)
        ref = torch.fake_quantize_per_tensor_affine(x, 0.1, zp, qmin, qmax)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_fq_per_tensor_affine_quant_min_equals_quant_max():
    x = torch.tensor([-5.0, 0.0, 5.0])
    out = fake_quantize_per_tensor_affine(x, 0.1, 5, 5, 5)
    ref = torch.fake_quantize_per_tensor_affine(x, 0.1, 5, 5, 5)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(out, torch.zeros(3), rtol=0, atol=0)


def test_fq_per_tensor_affine_extreme_scales():
    torch.manual_seed(2)
    x = torch.randn(2, 33)
    for scale in (1e-4, 16.0):
        out = fake_quantize_per_tensor_affine(x, scale, 0, -128, 127)
        ref = torch.fake_quantize_per_tensor_affine(x, scale, 0, -128, 127)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_fq_per_tensor_affine_int8_vs_uint8_ranges():
    torch.manual_seed(3)
    x = torch.randn(4, 8)
    out_i8 = fake_quantize_per_tensor_affine(x, 0.05, 0, -128, 127)
    out_u8 = fake_quantize_per_tensor_affine(x, 0.05, 128, 0, 255)
    ref_i8 = torch.fake_quantize_per_tensor_affine(x, 0.05, 0, -128, 127)
    ref_u8 = torch.fake_quantize_per_tensor_affine(x, 0.05, 128, 0, 255)
    torch.testing.assert_close(out_i8, ref_i8, rtol=0, atol=0)
    torch.testing.assert_close(out_u8, ref_u8, rtol=0, atol=0)


def test_fq_per_tensor_affine_float16():
    torch.manual_seed(4)
    x = torch.randn(2, 17, dtype=torch.float16)
    out = fake_quantize_per_tensor_affine(x, 0.1, 0, -128, 127)
    ref = torch.fake_quantize_per_tensor_affine(x, 0.1, 0, -128, 127)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_fq_per_tensor_affine_non_contiguous():
    torch.manual_seed(5)
    xt = torch.randn(4, 6).t()
    assert not xt.is_contiguous()
    out = fake_quantize_per_tensor_affine(xt, 0.1, 0, -128, 127)
    ref = torch.fake_quantize_per_tensor_affine(xt, 0.1, 0, -128, 127)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    xs = torch.randn(4, 8)[::2, ::2]
    out = fake_quantize_per_tensor_affine(xs, 0.1, 0, -128, 127)
    ref = torch.fake_quantize_per_tensor_affine(xs, 0.1, 0, -128, 127)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_fq_per_tensor_affine_tensor_valued_qparams():
    torch.manual_seed(6)
    x = torch.randn(5)
    out = fake_quantize_per_tensor_affine(
        x, torch.tensor(0.1), torch.tensor(3), torch.tensor(-128), torch.tensor(127)
    )
    ref = torch.fake_quantize_per_tensor_affine(x, 0.1, 3, -128, 127)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_fq_per_tensor_affine_empty_tensor():
    x = torch.empty(2, 0)
    out = fake_quantize_per_tensor_affine(x, 0.1, 0, -128, 127)
    assert out.shape == (2, 0)
    assert out.dtype == torch.float32


def test_fq_per_tensor_affine_errors():
    x = torch.randn(3)
    with pytest.raises(RuntimeError, match="quant_min must be less than or equal"):
        fake_quantize_per_tensor_affine(x, 0.1, 0, 5, -5)
    with pytest.raises(RuntimeError, match="zero_point must be between"):
        fake_quantize_per_tensor_affine(x, 0.1, 300, 0, 255)
    with pytest.raises(RuntimeError, match="zero_point must be between"):
        fake_quantize_per_tensor_affine(x, 0.1, -1, 0, 255)


def test_fq_per_tensor_affine_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.fake_quantize_per_tensor_affine")
    launches = []
    kernel = mod.fake_quantize_per_tensor_affine_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    x = torch.randn(3, 17)
    out = fake_quantize_per_tensor_affine(x, 0.1, 0, -128, 127)
    assert len(launches) >= 1
    torch.testing.assert_close(
        out, torch.fake_quantize_per_tensor_affine(x, 0.1, 0, -128, 127), rtol=0, atol=0
    )
