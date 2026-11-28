import importlib

import pytest
import torch

from .make_per_channel_quantized_tensor import _make_per_channel_quantized_tensor


def test_make_per_channel_int_dtypes_wrap_verbatim():
    torch.manual_seed(0)
    for int_dtype, qdtype in (
        (torch.int8, torch.qint8),
        (torch.uint8, torch.quint8),
        (torch.int32, torch.qint32),
    ):
        x = torch.randint(
            -100 if int_dtype != torch.uint8 else 0, 101, (3, 17), dtype=int_dtype
        )
        scales = torch.tensor([0.1, 0.2, 0.3])
        zps = torch.tensor([1, 2, 3], dtype=torch.int64)
        for axis in (0, 1):
            sc = scales if axis == 0 else torch.tensor([0.1] * 17)
            zp = zps if axis == 0 else torch.zeros(17, dtype=torch.int64)
            q = _make_per_channel_quantized_tensor(x, sc, zp, axis)
            assert q.is_quantized
            assert q.dtype == qdtype
            assert q.qscheme() == torch.per_channel_affine
            assert q.q_per_channel_axis() == axis
            torch.testing.assert_close(q.int_repr(), x, rtol=0, atol=0)
            torch.testing.assert_close(
                q.q_per_channel_scales().float(), sc.float(), rtol=0, atol=0
            )


def test_make_per_channel_axis_variations_and_negative():
    torch.manual_seed(1)
    x = torch.randint(-128, 128, (2, 3, 4), dtype=torch.int8)
    for axis, n in ((0, 2), (1, 3), (2, 4)):
        sc = torch.full((n,), 0.1)
        zp = torch.zeros(n, dtype=torch.int64)
        q = _make_per_channel_quantized_tensor(x, sc, zp, axis)
        assert q.q_per_channel_axis() == axis
        torch.testing.assert_close(q.int_repr(), x, rtol=0, atol=0)
    # the axis is stored verbatim, negative values included
    sc = torch.full((4,), 0.1)
    zp = torch.zeros(4, dtype=torch.int64)
    q = _make_per_channel_quantized_tensor(x, sc, zp, -1)
    assert q.q_per_channel_axis() == -1


def test_make_per_channel_dequant_formula():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)
    scales = torch.tensor([0.1, 0.2])
    zps = torch.tensor([1, 2], dtype=torch.int64)
    q = _make_per_channel_quantized_tensor(x, scales, zps, 0)
    # axis=0: qparams apply per row (column vector), like the aten
    # dequantize of this tensor (verified: matches torch.dequantize of the
    # aten-constructed ref bit-exactly).
    expected = (x.to(torch.float32) - zps.to(torch.float32).reshape(-1, 1)) * (
        scales.to(torch.float32).reshape(-1, 1)
    )
    torch.testing.assert_close(torch.dequantize(q), expected, rtol=0, atol=0)


def test_make_per_channel_float_zero_points_select_float_qparams():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)
    scales = torch.tensor([0.1, 0.2])
    zps = torch.tensor([0.5, 1.5])
    q = _make_per_channel_quantized_tensor(x, scales, zps, 0)
    assert q.qscheme() == torch.per_channel_affine_float_qparams
    torch.testing.assert_close(q.int_repr(), x, rtol=0, atol=0)
    torch.testing.assert_close(
        q.q_per_channel_zero_points().float(), zps, rtol=0, atol=0
    )
    ref = torch._make_per_channel_quantized_tensor(x, scales, zps, 0)
    assert ref.qscheme() == torch.per_channel_affine_float_qparams
    torch.testing.assert_close(
        torch.dequantize(q), torch.dequantize(ref), rtol=0, atol=0
    )


def test_make_per_channel_matches_aten_and_quantize_equivalence():
    torch.manual_seed(2)
    x = torch.randint(-128, 128, (4, 6), dtype=torch.int8)
    scales = torch.tensor([0.1, 0.2, 0.3, 0.4])
    zps = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    q = _make_per_channel_quantized_tensor(x, scales, zps, 0)
    ref = torch._make_per_channel_quantized_tensor(x, scales, zps, 0)
    assert q.dtype == ref.dtype
    assert q.q_per_channel_axis() == ref.q_per_channel_axis()
    torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.dequantize(q), torch.dequantize(ref), rtol=0, atol=0
    )
    # cross-check with torch.quantize_per_channel on constructed cases
    y = torch.randn(4, 6)
    qy = torch.quantize_per_channel(y, scales, zps, 0, torch.qint8)
    q2 = _make_per_channel_quantized_tensor(qy.int_repr(), scales, zps, 0)
    torch.testing.assert_close(
        torch.dequantize(q2), torch.dequantize(qy), rtol=0, atol=0
    )


def test_make_per_channel_float32_input_quantizes_to_qint8():
    torch.manual_seed(3)
    x = torch.randn(3, 17)
    scales = torch.tensor([0.1, 0.2, 0.3])
    zps = torch.tensor([1, 2, 3], dtype=torch.int64)
    q = _make_per_channel_quantized_tensor(x, scales, zps, 0)
    assert q.dtype == torch.qint8
    expected = (
        (torch.round(x / scales.reshape(-1, 1)) + zps.to(torch.float32).reshape(-1, 1))
        .clamp(-128, 127)
        .to(torch.int8)
    )
    torch.testing.assert_close(q.int_repr(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.dequantize(q),
        (expected.to(torch.float32) - zps.to(torch.float32).reshape(-1, 1))
        * scales.to(torch.float32).reshape(-1, 1),
        rtol=0,
        atol=0,
    )


def test_make_per_channel_float16_input_quantizes_to_quint8():
    torch.manual_seed(4)
    x = torch.randn(2, 9, dtype=torch.float16)
    scales = torch.tensor([0.1, 0.2])
    zps = torch.tensor([0, 5], dtype=torch.int64)
    q = _make_per_channel_quantized_tensor(x, scales, zps, 0)
    assert q.dtype == torch.quint8
    expected = (
        (
            torch.round(x.float() / scales.reshape(-1, 1).float())
            + zps.to(torch.float32).reshape(-1, 1)
        )
        .clamp(0, 255)
        .to(torch.uint8)
    )
    torch.testing.assert_close(q.int_repr(), expected, rtol=0, atol=0)


def test_make_per_channel_non_contiguous_input():
    torch.manual_seed(5)
    base = torch.arange(24, dtype=torch.int8).reshape(4, 6)
    for x in (base.t(), base[::2, ::2]):
        sc = torch.full((x.shape[0],), 0.1)
        zp = torch.zeros(x.shape[0], dtype=torch.int64)
        q = _make_per_channel_quantized_tensor(x, sc, zp, 0)
        torch.testing.assert_close(q.int_repr(), x, rtol=0, atol=0)


def test_make_per_channel_empty_tensor():
    q = _make_per_channel_quantized_tensor(
        torch.empty(0, 4, dtype=torch.int8),
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        torch.zeros(4, dtype=torch.int64),
        0,
    )
    assert q.int_repr().shape == (0, 4)


def test_make_per_channel_unsupported_dtype():
    with pytest.raises(TypeError, match="Only int8/uint8/int32/float32/float16"):
        _make_per_channel_quantized_tensor(
            torch.zeros(3, dtype=torch.int64),
            torch.tensor([0.1]),
            torch.tensor([0], dtype=torch.int64),
            0,
        )


def test_make_per_channel_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.make_per_channel_quantized_tensor")
    cases = (
        (
            "_pcq_copy_kernel",
            torch.randint(-128, 128, (3, 17), dtype=torch.int8),
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([1, 2, 3], dtype=torch.int64),
        ),
        (
            "_pcq_quant_kernel",
            torch.randn(3, 17),
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([1, 2, 3], dtype=torch.int64),
        ),
    )
    for kernel_name, x, sc, zp in cases:
        launches = []
        kernel = getattr(mod, kernel_name)
        real_run = kernel.run

        def spy_run(*args, _real=real_run, **kwargs):
            launches.append(kwargs.get("grid"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
        q = _make_per_channel_quantized_tensor(x, sc, zp, 0)
        assert len(launches) >= 1, f"{kernel_name} must be launched"
        assert q.dtype in (torch.qint8, torch.quint8)
        assert q.int_repr().shape == x.shape
