import importlib

import torch

from ._fake_quantize_per_tensor_affine_cachemask_tensor_qparams import (
    _fake_quantize_per_tensor_affine_cachemask_tensor_qparams,
)


def _ref(x, scale, zp, enabled, qmin, qmax):
    return torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        x,
        torch.tensor([scale]),
        torch.tensor([zp]),
        torch.tensor([int(enabled)], dtype=torch.int32),
        qmin,
        qmax,
    )


def test_tensor_qparams_returns_out_and_bool_mask():
    torch.manual_seed(0)
    x = torch.randn(2, 3)
    result = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        x,
        torch.tensor([0.1]),
        torch.tensor([0.0]),
        torch.tensor([1], dtype=torch.int32),
        -128,
        127,
    )
    assert isinstance(result, tuple) and len(result) == 2
    out, mask = result
    ref_out, ref_mask = _ref(x, 0.1, 0.0, True, -128, 127)
    assert mask.dtype == torch.bool
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_tensor_qparams_matches_aten_shapes_and_ranges():
    torch.manual_seed(1)
    for shape in ((7,), (3, 17), (2, 3, 5)):
        x = torch.randn(shape)
        for scale, zp, qmin, qmax in (
            (0.1, 0.0, -128, 127),
            (0.05, 128.0, 0, 255),
            (1e-4, 3.0, -128, 127),
            (16.0, 0.0, 0, 255),
        ):
            out, mask = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
                x,
                torch.tensor([scale]),
                torch.tensor([zp]),
                torch.tensor([1], dtype=torch.int32),
                qmin,
                qmax,
            )
            ref_out, ref_mask = _ref(x, scale, zp, True, qmin, qmax)
            torch.testing.assert_close(out, ref_out, rtol=0, atol=0, msg=f"{shape}")
            torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_tensor_qparams_mask_semantics_in_range_pre_clamp():
    x = torch.tensor([0.04, 100.0, -100.0, 0.05])
    out, mask = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        x,
        torch.tensor([0.1]),
        torch.tensor([0.0]),
        torch.tensor([1], dtype=torch.int32),
        0,
        255,
    )
    ref_out, ref_mask = _ref(x, 0.1, 0.0, True, 0, 255)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)
    assert mask.tolist() == [True, False, False, True]


def test_tensor_qparams_disabled_bit_exact_copy_and_mask_all_true():
    torch.manual_seed(2)
    x = torch.randn(3, 5)
    out, mask = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        x,
        torch.tensor([0.1]),
        torch.tensor([0.0]),
        torch.tensor([0], dtype=torch.int32),
        -128,
        127,
    )
    assert torch.equal(out, x)
    assert mask.dtype == torch.bool
    assert bool(mask.all())
    # int64 enabled flag also selects disabled (reinterpreted by the wrapper)
    out64, mask64 = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        x,
        torch.tensor([0.1]),
        torch.tensor([0.0]),
        torch.tensor([0], dtype=torch.int64),
        -128,
        127,
    )
    assert torch.equal(out64, x)
    assert bool(mask64.all())


def test_tensor_qparams_int64_enabled_flag_quantizes():
    torch.manual_seed(3)
    x = torch.randn(2, 9)
    out, mask = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        x,
        torch.tensor([0.1]),
        torch.tensor([0.0]),
        torch.tensor([1], dtype=torch.int64),
        -128,
        127,
    )
    ref_out, ref_mask = _ref(x, 0.1, 0.0, True, -128, 127)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_tensor_qparams_nan_and_saturation():
    x = torch.tensor([float("nan"), 1.0, -1.0, 100.0])
    out, mask = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        x,
        torch.tensor([0.1]),
        torch.tensor([0.0]),
        torch.tensor([1], dtype=torch.int32),
        -128,
        127,
    )
    ref_out, ref_mask = _ref(x, 0.1, 0.0, True, -128, 127)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)
    # NaN clamps to quant_min: (qmin - zp) * scale = -12.8, mask False
    assert mask.tolist() == [False, True, True, False]
    torch.testing.assert_close(out[0], torch.tensor(-12.8), rtol=0, atol=0)


def test_tensor_qparams_fractional_zero_point():
    x = torch.tensor([0.25, 0.26, 1.0])
    out, mask = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        x,
        torch.tensor([0.1]),
        torch.tensor([2.5]),
        torch.tensor([1], dtype=torch.int32),
        -128,
        127,
    )
    ref_out, ref_mask = _ref(x, 0.1, 2.5, True, -128, 127)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_tensor_qparams_zero_point_edges_and_degenerate_range():
    torch.manual_seed(4)
    x = torch.randn(2, 9)
    for zp, qmin, qmax in ((-128.0, -128, 127), (127.0, -128, 127), (0.0, 0, 255)):
        out, mask = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
            x,
            torch.tensor([0.1]),
            torch.tensor([zp]),
            torch.tensor([1], dtype=torch.int32),
            qmin,
            qmax,
        )
        ref_out, ref_mask = _ref(x, 0.1, zp, True, qmin, qmax)
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
        torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)
    out, mask = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        x,
        torch.tensor([0.1]),
        torch.tensor([5.0]),
        torch.tensor([1], dtype=torch.int32),
        5,
        5,
    )
    ref_out, ref_mask = _ref(x, 0.1, 5.0, True, 5, 5)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    # q == 5 == zp is IN the degenerate [5, 5] range; the exact set of
    # in-range lanes is pinned by the oracle comparison (aten's own mask has
    # True lanes for x rounding to 0 here).
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_tensor_qparams_float16():
    torch.manual_seed(5)
    x = torch.randn(2, 17, dtype=torch.float16)
    out, mask = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        x,
        torch.tensor([0.1]),
        torch.tensor([0.0]),
        torch.tensor([1], dtype=torch.int32),
        -128,
        127,
    )
    ref_out, ref_mask = _ref(x, 0.1, 0.0, True, -128, 127)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)
    out, mask = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        x,
        torch.tensor([0.1]),
        torch.tensor([0.0]),
        torch.tensor([0], dtype=torch.int32),
        -128,
        127,
    )
    assert torch.equal(out, x)


def test_tensor_qparams_non_contiguous():
    torch.manual_seed(6)
    xt = torch.randn(4, 6).t()
    out, mask = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        xt,
        torch.tensor([0.1]),
        torch.tensor([0.0]),
        torch.tensor([1], dtype=torch.int32),
        -128,
        127,
    )
    ref_out, ref_mask = _ref(xt, 0.1, 0.0, True, -128, 127)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)
    xs = torch.randn(4, 8)[::2, ::2]
    out, mask = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        xs,
        torch.tensor([0.1]),
        torch.tensor([0.0]),
        torch.tensor([1], dtype=torch.int32),
        -128,
        127,
    )
    ref_out, ref_mask = _ref(xs, 0.1, 0.0, True, -128, 127)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_tensor_qparams_empty_tensor():
    out, mask = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        torch.empty(2, 0),
        torch.tensor([0.1]),
        torch.tensor([0.0]),
        torch.tensor([1], dtype=torch.int32),
        -128,
        127,
    )
    assert out.shape == (2, 0) and mask.shape == (2, 0)
    assert mask.dtype == torch.bool


def test_tensor_qparams_triton_path(monkeypatch):
    mod = importlib.import_module(
        "flaggems._fake_quantize_per_tensor_affine_cachemask_tensor_qparams"
    )
    x = torch.randn(3, 17)
    xs = torch.randn(4, 8)[::2, ::2]
    for kernel_name, inp in (("_fq_flat_kernel", x), ("_fq_strided_kernel", xs)):
        launches = []
        kernel = getattr(mod, kernel_name)
        real_run = kernel.run

        def spy_run(*args, _real=real_run, **kwargs):
            launches.append(kwargs.get("grid"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
        out, mask = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
            inp,
            torch.tensor([0.1]),
            torch.tensor([0.0]),
            torch.tensor([1], dtype=torch.int32),
            -128,
            127,
        )
        assert len(launches) >= 1, f"{kernel_name} must be launched"
        ref_out, ref_mask = _ref(inp, 0.1, 0.0, True, -128, 127)
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
        torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)
