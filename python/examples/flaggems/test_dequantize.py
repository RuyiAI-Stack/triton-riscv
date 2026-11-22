import importlib

import pytest
import torch

from .dequantize import dequantize


def _make_qtensor(values, scale, zero_point, qdtype):
    return torch.quantize_per_tensor(
        torch.tensor(values, dtype=torch.float32), scale, zero_point, qdtype
    )


def test_dequantize_qint8_matches_aten():
    torch.manual_seed(0)
    for shape in ((7,), (3, 17), (2, 3, 5)):
        x = torch.randn(shape)
        for scale, zp in ((0.1, 5), (0.03125, -3), (2.0, 0)):
            q = torch.quantize_per_tensor(x, scale, zp, torch.qint8)
            out = dequantize(q)
            ref = torch.dequantize(q)
            assert out.dtype == torch.float32
            torch.testing.assert_close(out, ref, rtol=0, atol=0)
            assert out.shape == ref.shape


def test_dequantize_quint8_and_qint32():
    torch.manual_seed(1)
    x = torch.randn(4, 6)
    q8u = torch.quantize_per_tensor(x, 0.05, 128, torch.quint8)
    out = dequantize(q8u)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.dequantize(q8u), rtol=0, atol=0)
    q32 = torch.quantize_per_tensor(x, 0.01, 2, torch.qint32)
    out32 = dequantize(q32)
    assert out32.dtype == torch.float32
    torch.testing.assert_close(out32, torch.dequantize(q32), rtol=0, atol=0)


def test_dequantize_formula_int_repr_minus_zp_times_scale():
    q = _make_qtensor([1.0, -2.0, 3.5, 0.0], 0.1, 5, torch.qint8)
    out = dequantize(q)
    expected = (q.int_repr().to(torch.float32) - 5) * 0.1
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_dequantize_saturation_extremes():
    q = _make_qtensor([1e6, -1e6, 0.0], 0.1, 128, torch.quint8)
    out = dequantize(q)
    torch.testing.assert_close(out, torch.dequantize(q), rtol=0, atol=0)
    # clamped int_repr [255, 0, 128] dequantizes to the clamped values
    torch.testing.assert_close(out, torch.tensor([12.7, -12.8, 0.0]), rtol=0, atol=0)


def test_dequantize_small_and_large_scale():
    for scale in (1e-4, 64.0):
        q = _make_qtensor([1.0, -1.0, 0.5], scale, 7, torch.qint8)
        torch.testing.assert_close(dequantize(q), torch.dequantize(q), rtol=0, atol=0)


def test_dequantize_non_contiguous_quantized_input():
    torch.manual_seed(2)
    q = torch.quantize_per_tensor(torch.randn(10), 0.1, 3, torch.qint8)
    qs = q[::2]
    assert qs.int_repr().shape == (5,)
    out = dequantize(qs)
    torch.testing.assert_close(out, torch.dequantize(qs), rtol=0, atol=0)


def test_dequantize_empty_tensor():
    q = torch.quantize_per_tensor(torch.empty(3, 0), 0.1, 3, torch.qint8)
    out = dequantize(q)
    assert out.shape == (3, 0)
    assert out.dtype == torch.float32


def test_dequantize_rejects_per_channel_quantized_tensor():
    q = torch.quantize_per_channel(
        torch.randn(3, 4),
        torch.tensor([0.1, 0.2, 0.3]),
        torch.tensor([1, 2, 3], dtype=torch.int64),
        0,
        torch.qint8,
    )
    with pytest.raises(RuntimeError, match="kPerTensorAffine"):
        dequantize(q)


def test_dequantize_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.dequantize")
    launches = []
    kernel = mod.dequantize_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    q = torch.quantize_per_tensor(torch.randn(3, 17), 0.1, 5, torch.qint8)
    out = dequantize(q)
    assert len(launches) >= 1
    torch.testing.assert_close(out, torch.dequantize(q), rtol=0, atol=0)
