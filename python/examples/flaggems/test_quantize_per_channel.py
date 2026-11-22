import importlib

import pytest
import torch

from .quantize_per_channel import quantize_per_channel, quantize_per_channel_out


def test_quantize_per_channel_matches_aten_axis0():
    torch.manual_seed(0)
    x = torch.randn(3, 17)
    scales = torch.tensor([0.1, 0.2, 0.3])
    zps = torch.tensor([1, 2, 3], dtype=torch.int64)
    q = quantize_per_channel(x, scales, zps, 0, torch.qint8)
    ref = torch.quantize_per_channel(x, scales, zps, 0, torch.qint8)
    assert q.dtype == torch.qint8
    assert q.qscheme() == torch.per_channel_affine
    assert q.q_per_channel_axis() == 0
    torch.testing.assert_close(
        q.q_per_channel_scales(), scales.double(), rtol=0, atol=0
    )
    torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.dequantize(q), torch.dequantize(ref), rtol=0, atol=0
    )


def test_quantize_per_channel_axis_variations():
    torch.manual_seed(1)
    x3d = torch.randn(2, 3, 4)
    for axis, n in ((0, 2), (1, 3), (2, 4)):
        scales = torch.full((n,), 0.1)
        zps = torch.arange(n, dtype=torch.int64)
        q = quantize_per_channel(x3d, scales, zps, axis, torch.qint8)
        ref = torch.quantize_per_channel(x3d, scales, zps, axis, torch.qint8)
        assert q.q_per_channel_axis() == axis
        torch.testing.assert_close(
            q.int_repr(), ref.int_repr(), rtol=0, atol=0, msg=f"axis={axis}"
        )


def test_quantize_per_channel_qint8_quint8_qint32():
    torch.manual_seed(2)
    x = torch.randn(4, 8)
    scales = torch.tensor([0.05, 0.1, 0.2, 0.3])
    for qdtype, zp_vals, storage_dtype in (
        (torch.qint8, [-128, 0, 5, 127], torch.int8),
        (torch.quint8, [0, 128, 200, 255], torch.uint8),
        (torch.qint32, [0, 1, 2, 3], torch.int32),
    ):
        zps = torch.tensor(zp_vals, dtype=torch.int64)
        q = quantize_per_channel(x, scales, zps, 0, qdtype)
        ref = torch.quantize_per_channel(x, scales, zps, 0, qdtype)
        assert q.dtype == qdtype
        assert q.int_repr().dtype == storage_dtype
        torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)


def test_quantize_per_channel_saturation_and_extreme_scales():
    x = torch.tensor([[1e6, -1e6, 0.05, -0.05]])
    scales = torch.tensor([0.1])
    zps = torch.tensor([0], dtype=torch.int64)
    q = quantize_per_channel(x, scales, zps, 0, torch.qint8)
    ref = torch.quantize_per_channel(x, scales, zps, 0, torch.qint8)
    torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)
    torch.manual_seed(3)
    xr = torch.randn(3, 9)
    for scales in (torch.full((3,), 1e-4), torch.full((3,), 32.0)):
        zps = torch.zeros(3, dtype=torch.int64)
        q = quantize_per_channel(xr, scales, zps, 0, torch.qint8)
        ref = torch.quantize_per_channel(xr, scales, zps, 0, torch.qint8)
        torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)


def test_quantize_per_channel_float_zero_points_scheme():
    torch.manual_seed(4)
    x = torch.randn(3, 4)
    scales = torch.tensor([0.1, 0.2, 0.3])
    zps = torch.tensor([0.5, 1.5, 2.5])
    q = quantize_per_channel(x, scales, zps, 0, torch.qint8)
    ref = torch.quantize_per_channel(x, scales, zps, 0, torch.qint8)
    assert q.qscheme() == torch.per_channel_affine_float_qparams
    torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)
    torch.testing.assert_close(
        q.q_per_channel_zero_points().float(), zps, rtol=0, atol=0
    )
    torch.testing.assert_close(
        torch.dequantize(q), torch.dequantize(ref), rtol=0, atol=0
    )


def test_quantize_per_channel_non_contiguous_input():
    torch.manual_seed(5)
    xt = torch.randn(4, 6).t()
    scales = torch.full((6,), 0.1)
    zps = torch.zeros(6, dtype=torch.int64)
    q = quantize_per_channel(xt, scales, zps, 0, torch.qint8)
    ref = torch.quantize_per_channel(xt, scales, zps, 0, torch.qint8)
    torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)


def test_quantize_per_channel_empty():
    # A 0-size channel axis demands 0-length qparams: aten and the wrapper's
    # channel-count validation reject 4 scales against size(axis)=0.
    q = quantize_per_channel(
        torch.empty(0, 4),
        torch.empty(0),
        torch.zeros(0, dtype=torch.int64),
        0,
        torch.qint8,
    )
    assert q.int_repr().shape == (0, 4)


def test_quantize_per_channel_errors():
    x = torch.randn(3, 4)
    scales = torch.tensor([0.1, 0.2, 0.3])
    zps = torch.tensor([1, 2, 3], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="Channel axis out of range"):
        quantize_per_channel(x, scales, zps, 7, torch.qint8)
    with pytest.raises(RuntimeError, match="length of scales must equal to channel"):
        quantize_per_channel(
            x,
            torch.tensor([0.1, 0.2]),
            torch.tensor([1, 2], dtype=torch.int64),
            0,
            torch.qint8,
        )
    with pytest.raises(RuntimeError, match="expects a Float Tensor, got Double"):
        quantize_per_channel(x.double(), scales, zps, 0, torch.qint8)
    with pytest.raises(RuntimeError, match="zero_point is below lower bound"):
        quantize_per_channel(
            x, scales, torch.tensor([-200, 2, 3], dtype=torch.int64), 0, torch.qint8
        )
    with pytest.raises(RuntimeError, match="zero_points tensor must have dimension 1"):
        quantize_per_channel(
            x, scales, torch.tensor([[1, 2, 3]], dtype=torch.int64), 0, torch.qint8
        )
    with pytest.raises(NotImplementedError, match="not implemented"):
        quantize_per_channel(x, scales, zps, 0, torch.float32)


def test_quantize_per_channel_out_writes_in_place():
    torch.manual_seed(6)
    x = torch.randn(3, 5)
    scales = torch.tensor([0.1, 0.2, 0.3])
    zps = torch.tensor([1, 2, 3], dtype=torch.int64)
    out = torch.quantize_per_channel(
        torch.zeros(3, 5),
        torch.tensor([0.5, 0.5, 0.5]),
        torch.zeros(3, dtype=torch.int64),
        0,
        torch.qint8,
    )
    ret = quantize_per_channel_out(x, scales, zps, 0, torch.qint8, out=out)
    assert ret is out
    ref = torch.quantize_per_channel(x, scales, zps, 0, torch.qint8)
    torch.testing.assert_close(out.int_repr(), ref.int_repr(), rtol=0, atol=0)
    assert out.q_per_channel_axis() == 0
    torch.testing.assert_close(
        torch.dequantize(out), torch.dequantize(ref), rtol=0, atol=0
    )


def test_quantize_per_channel_out_none_behaves_functional():
    torch.manual_seed(7)
    x = torch.randn(3, 5)
    scales = torch.tensor([0.1, 0.2, 0.3])
    zps = torch.tensor([1, 2, 3], dtype=torch.int64)
    q = quantize_per_channel_out(x, scales, zps, 0, torch.qint8)
    ref = torch.quantize_per_channel(x, scales, zps, 0, torch.qint8)
    torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)


def test_quantize_per_channel_out_errors():
    x = torch.randn(3, 4)
    scales = torch.tensor([0.1, 0.2, 0.3])
    zps = torch.tensor([1, 2, 3], dtype=torch.int64)
    out = torch.quantize_per_channel(torch.zeros(3, 4), scales, zps, 0, torch.quint8)
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        quantize_per_channel_out(x, scales, zps, 0, torch.qint8, out=out)
    out_f = torch.quantize_per_channel(
        torch.zeros(3, 4), scales, torch.tensor([0.5, 1.5, 2.5]), 0, torch.qint8
    )
    with pytest.raises(RuntimeError, match="same qscheme"):
        quantize_per_channel_out(x, scales, zps, 0, torch.qint8, out=out_f)


def test_quantize_per_channel_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.quantize_per_channel")
    launches = []
    kernel = mod.quantize_per_channel_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    x = torch.randn(3, 17)
    scales = torch.tensor([0.1, 0.2, 0.3])
    zps = torch.tensor([1, 2, 3], dtype=torch.int64)
    q = quantize_per_channel(x, scales, zps, 0, torch.qint8)
    assert len(launches) >= 1
    ref = torch.quantize_per_channel(x, scales, zps, 0, torch.qint8)
    torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)
