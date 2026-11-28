import importlib

import pytest
import torch

from .fake_quantize_per_tensor_affine_cachemask import (
    fake_quantize_per_tensor_affine_cachemask,
    fake_quantize_per_tensor_affine_cachemask_out,
)


def _ref(x, scale, zp, qmin, qmax):
    return torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
        x, scale, zp, qmin, qmax
    )


def test_cachemask_returns_out_and_bool_mask():
    torch.manual_seed(0)
    x = torch.randn(2, 3)
    result = fake_quantize_per_tensor_affine_cachemask(x, 0.1, 0, -128, 127)
    assert isinstance(result, tuple) and len(result) == 2
    out, mask = result
    ref_out, ref_mask = _ref(x, 0.1, 0, -128, 127)
    assert mask.dtype == torch.bool
    assert out.dtype == x.dtype
    assert out.shape == x.shape and mask.shape == x.shape
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_cachemask_matches_aten_shapes_and_ranges():
    torch.manual_seed(1)
    for shape in ((7,), (3, 17), (2, 3, 5)):
        x = torch.randn(shape)
        for scale, zp, qmin, qmax in (
            (0.1, 0, -128, 127),
            (0.05, 128, 0, 255),
            (1e-4, 3, -128, 127),
            (16.0, 0, 0, 255),
        ):
            out, mask = fake_quantize_per_tensor_affine_cachemask(
                x, scale, zp, qmin, qmax
            )
            ref_out, ref_mask = _ref(x, scale, zp, qmin, qmax)
            torch.testing.assert_close(out, ref_out, rtol=0, atol=0, msg=f"{shape}")
            torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_cachemask_mask_semantics_in_range_pre_clamp():
    # mask is True exactly where the rounded quantized value (before clamping)
    # stays inside [quant_min, quant_max]; saturated elements report False.
    x = torch.tensor([0.04, 100.0, -100.0, 0.05, -0.05])
    out, mask = fake_quantize_per_tensor_affine_cachemask(x, 0.1, 0, 0, 255)
    ref_out, ref_mask = _ref(x, 0.1, 0, 0, 255)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)
    assert mask.tolist() == [True, False, False, True, True]


def test_cachemask_saturation_values():
    x = torch.tensor([1000.0, -1000.0, 0.05])
    out, mask = fake_quantize_per_tensor_affine_cachemask(x, 0.1, 0, 0, 255)
    torch.testing.assert_close(out, torch.tensor([25.5, 0.0, 0.0]), rtol=0, atol=0)
    assert mask.tolist() == [False, False, True]


def test_cachemask_nan_input():
    x = torch.tensor([float("nan"), 1.0, -1.0])
    out, mask = fake_quantize_per_tensor_affine_cachemask(x, 0.1, 0, 0, 255)
    ref_out, ref_mask = _ref(x, 0.1, 0, 0, 255)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)
    # -1.0 with scale 0.1 rounds to q = -10, outside [0, 255] (probed aten mask)
    assert mask.tolist() == [False, True, False]


def test_cachemask_zero_point_at_edges_and_degenerate_range():
    torch.manual_seed(2)
    x = torch.randn(2, 9)
    for zp, qmin, qmax in ((-128, -128, 127), (127, -128, 127), (0, 0, 255)):
        out, mask = fake_quantize_per_tensor_affine_cachemask(x, 0.1, zp, qmin, qmax)
        ref_out, ref_mask = _ref(x, 0.1, zp, qmin, qmax)
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
        torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)
    # quant_min == quant_max == zero_point: out is all zeros; elements rounding
    # exactly to zero_point stay in range, so the mask is NOT all False (probed aten)
    out, mask = fake_quantize_per_tensor_affine_cachemask(x, 0.1, 5, 5, 5)
    ref_out, ref_mask = _ref(x, 0.1, 5, 5, 5)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)


def test_cachemask_float16():
    torch.manual_seed(3)
    x = torch.randn(2, 17, dtype=torch.float16)
    out, mask = fake_quantize_per_tensor_affine_cachemask(x, 0.1, 0, -128, 127)
    ref_out, ref_mask = _ref(x, 0.1, 0, -128, 127)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_cachemask_non_contiguous():
    torch.manual_seed(4)
    xt = torch.randn(4, 6).t()
    out, mask = fake_quantize_per_tensor_affine_cachemask(xt, 0.1, 0, -128, 127)
    ref_out, ref_mask = _ref(xt, 0.1, 0, -128, 127)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)
    xs = torch.randn(4, 8)[::2, ::2]
    out, mask = fake_quantize_per_tensor_affine_cachemask(xs, 0.1, 0, -128, 127)
    ref_out, ref_mask = _ref(xs, 0.1, 0, -128, 127)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_cachemask_empty_tensor():
    x = torch.empty(2, 0)
    out, mask = fake_quantize_per_tensor_affine_cachemask(x, 0.1, 0, -128, 127)
    assert out.shape == (2, 0) and mask.shape == (2, 0)
    assert mask.dtype == torch.bool


def test_cachemask_errors():
    x = torch.randn(3)
    with pytest.raises(RuntimeError, match="quant_min must be less than or equal"):
        fake_quantize_per_tensor_affine_cachemask(x, 0.1, 0, 5, -5)
    with pytest.raises(RuntimeError, match="zero_point must be between"):
        fake_quantize_per_tensor_affine_cachemask(x, 0.1, 300, 0, 255)
    with pytest.raises(NotImplementedError, match="cachemask_kernel_type_handling"):
        fake_quantize_per_tensor_affine_cachemask(torch.arange(4), 0.1, 0, -128, 127)


def test_cachemask_out_writes_into_buffers():
    torch.manual_seed(5)
    x = torch.randn(2, 17)
    out0 = torch.empty(2, 17)
    out1 = torch.empty(2, 17, dtype=torch.bool)
    ret0, ret1 = fake_quantize_per_tensor_affine_cachemask_out(
        x, 0.1, 0, -128, 127, out0=out0, out1=out1
    )
    assert ret0 is out0 and ret1 is out1
    ref_out, ref_mask = _ref(x, 0.1, 0, -128, 127)
    torch.testing.assert_close(out0, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(out1, ref_mask, rtol=0, atol=0)


def test_cachemask_out_resizes_mismatched_buffers():
    torch.manual_seed(6)
    x = torch.randn(3, 5)
    out0 = torch.empty(7)
    out1 = torch.empty(7, dtype=torch.bool)
    ret0, ret1 = fake_quantize_per_tensor_affine_cachemask_out(
        x, 0.1, 0, -128, 127, out0=out0, out1=out1
    )
    assert ret0.shape == (3, 5) and ret1.shape == (3, 5)
    ref_out, ref_mask = _ref(x, 0.1, 0, -128, 127)
    torch.testing.assert_close(ret0, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(ret1, ref_mask, rtol=0, atol=0)


def test_cachemask_out_rejects_wrong_dtypes():
    x = torch.randn(3)
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        fake_quantize_per_tensor_affine_cachemask_out(
            x,
            0.1,
            0,
            -128,
            127,
            out0=torch.empty(3, dtype=torch.float64),
            out1=torch.empty(3, dtype=torch.bool),
        )
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype bool"):
        fake_quantize_per_tensor_affine_cachemask_out(
            x,
            0.1,
            0,
            -128,
            127,
            out0=torch.empty(3),
            out1=torch.empty(3, dtype=torch.uint8),
        )


def test_cachemask_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.fake_quantize_per_tensor_affine_cachemask")
    x = torch.randn(3, 17)
    xs = torch.randn(4, 8)[::2, ::2]
    for kernel_name, inp in (
        ("fake_quantize_per_tensor_affine_cachemask_kernel", x),
        ("_fq_strided_kernel", xs),
    ):
        launches = []
        kernel = getattr(mod, kernel_name)
        real_run = kernel.run

        def spy_run(*args, _real=real_run, **kwargs):
            launches.append(kwargs.get("grid"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
        out, mask = fake_quantize_per_tensor_affine_cachemask(inp, 0.1, 0, -128, 127)
        assert len(launches) >= 1, f"{kernel_name} must be launched"
        ref_out, ref_mask = _ref(inp, 0.1, 0, -128, 127)
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
        torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)
