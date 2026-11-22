import importlib

import pytest
import torch

from .quantize_per_tensor import quantize_per_tensor, quantize_per_tensor_out


def test_quantize_per_tensor_int_repr_matches_aten():
    torch.manual_seed(0)
    for shape in ((7,), (3, 17), (2, 3, 5)):
        x = torch.randn(shape)
        for scale, zp in ((0.1, 5), (0.03125, -3), (2.0, 0)):
            q = quantize_per_tensor(x, scale, zp, torch.qint8)
            ref = torch.quantize_per_tensor(x, scale, zp, torch.qint8)
            assert q.is_quantized
            assert q.dtype == torch.qint8
            assert q.q_scale() == scale and q.q_zero_point() == zp
            assert q.qscheme() == torch.per_tensor_affine
            torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)
            torch.testing.assert_close(
                torch.dequantize(q), torch.dequantize(ref), rtol=0, atol=0
            )


def test_quantize_per_tensor_qint8_quint8_qint32():
    torch.manual_seed(1)
    x = torch.randn(4, 6)
    q_i8 = quantize_per_tensor(x, 0.05, 0, torch.qint8)
    q_u8 = quantize_per_tensor(x, 0.05, 128, torch.quint8)
    q_i32 = quantize_per_tensor(x, 0.01, 2, torch.qint32)
    assert q_i8.int_repr().dtype == torch.int8
    assert q_u8.int_repr().dtype == torch.uint8
    assert q_i32.int_repr().dtype == torch.int32
    torch.testing.assert_close(
        q_i8.int_repr(),
        torch.quantize_per_tensor(x, 0.05, 0, torch.qint8).int_repr(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        q_u8.int_repr(),
        torch.quantize_per_tensor(x, 0.05, 128, torch.quint8).int_repr(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        q_i32.int_repr(),
        torch.quantize_per_tensor(x, 0.01, 2, torch.qint32).int_repr(),
        rtol=0,
        atol=0,
    )


def test_quantize_per_tensor_saturation_and_zero_point_edges():
    x = torch.tensor([1e6, -1e6, 0.0])
    q = quantize_per_tensor(x, 0.1, 128, torch.quint8)
    ref = torch.quantize_per_tensor(x, 0.1, 128, torch.quint8)
    torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)
    torch.testing.assert_close(
        q.int_repr(), torch.tensor([255, 0, 128], dtype=torch.uint8), rtol=0, atol=0
    )
    qi = quantize_per_tensor(torch.tensor([1e6, -1e6]), 0.1, 0, torch.qint32)
    torch.testing.assert_close(
        qi.int_repr(),
        torch.tensor([10000000, -10000000], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.manual_seed(2)
    xr = torch.randn(2, 9)
    for zp, qdtype in (
        (-128, torch.qint8),
        (127, torch.qint8),
        (0, torch.quint8),
        (255, torch.quint8),
    ):
        q = quantize_per_tensor(xr, 0.1, zp, qdtype)
        ref = torch.quantize_per_tensor(xr, 0.1, zp, qdtype)
        torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)


def test_quantize_per_tensor_extreme_scales():
    torch.manual_seed(3)
    x = torch.randn(2, 33)
    for scale in (1e-4, 64.0):
        q = quantize_per_tensor(x, scale, 0, torch.qint8)
        ref = torch.quantize_per_tensor(x, scale, 0, torch.qint8)
        torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)


def test_quantize_per_tensor_non_contiguous_input():
    torch.manual_seed(4)
    xt = torch.randn(4, 6).t()
    q = quantize_per_tensor(xt, 0.1, 3, torch.qint8)
    ref = torch.quantize_per_tensor(xt, 0.1, 3, torch.qint8)
    torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)
    xs = torch.randn(4, 8)[::2, ::2]
    q = quantize_per_tensor(xs, 0.1, 3, torch.qint8)
    ref = torch.quantize_per_tensor(xs, 0.1, 3, torch.qint8)
    torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)


def test_quantize_per_tensor_channels_last_4d():
    torch.manual_seed(5)
    x = torch.randn(2, 3, 4, 5).to(memory_format=torch.channels_last)
    q = quantize_per_tensor(x, 0.1, 3, torch.qint8)
    ref = torch.quantize_per_tensor(x, 0.1, 3, torch.qint8)
    torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.dequantize(q), torch.dequantize(ref), rtol=0, atol=0
    )


def test_quantize_per_tensor_empty():
    q = quantize_per_tensor(torch.empty(3, 0), 0.1, 3, torch.qint8)
    assert q.int_repr().shape == (3, 0)
    assert torch.dequantize(q).shape == (3, 0)


def test_quantize_per_tensor_errors():
    x = torch.randn(3)
    with pytest.raises(RuntimeError, match="zero_point 300 is above upper bound"):
        quantize_per_tensor(x, 0.1, 300, torch.qint8)
    with pytest.raises(RuntimeError, match="zero_point -1 is below lower bound"):
        quantize_per_tensor(x, 0.1, -1, torch.quint8)
    with pytest.raises(RuntimeError, match="expected input of dtype torch.float32"):
        quantize_per_tensor(torch.randn(3, dtype=torch.float64), 0.1, 0, torch.qint8)
    with pytest.raises(RuntimeError, match="unsupported quantized dtype"):
        quantize_per_tensor(x, 0.1, 0, torch.float32)


def test_quantize_per_tensor_out_writes_in_place():
    torch.manual_seed(6)
    x = torch.randn(3, 5)
    out = torch.quantize_per_tensor(torch.zeros(3, 5), 0.5, 1, torch.qint8)
    ret = quantize_per_tensor_out(x, 0.1, 3, torch.qint8, out=out)
    assert ret is out
    ref = torch.quantize_per_tensor(x, 0.1, 3, torch.qint8)
    torch.testing.assert_close(out.int_repr(), ref.int_repr(), rtol=0, atol=0)
    # the quantizer is adopted from the call arguments
    assert out.q_scale() == 0.1 and out.q_zero_point() == 3
    torch.testing.assert_close(
        torch.dequantize(out), torch.dequantize(ref), rtol=0, atol=0
    )


def test_quantize_per_tensor_out_errors():
    x = torch.randn(3, 4)
    with pytest.raises(RuntimeError, match="argument 'out' is required"):
        quantize_per_tensor_out(x, 0.1, 3, torch.qint8)
    out = torch.quantize_per_tensor(torch.zeros(3, 4), 0.5, 1, torch.quint8)
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        quantize_per_tensor_out(x, 0.1, 3, torch.qint8, out=out)


def test_quantize_per_tensor_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.quantize_per_tensor")
    x = torch.randn(3, 17)
    xs = torch.randn(4, 8)[::2, ::2]
    for kernel_name, inp in (
        ("quantize_per_tensor_kernel", x),
        ("quantize_per_tensor_strided_kernel", xs),
    ):
        launches = []
        kernel = getattr(mod, kernel_name)
        real_run = kernel.run

        def spy_run(*args, _real=real_run, **kwargs):
            launches.append(kwargs.get("grid"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
        q = quantize_per_tensor(inp, 0.1, 3, torch.qint8)
        assert len(launches) >= 1, f"{kernel_name} must be launched"
        ref = torch.quantize_per_tensor(inp, 0.1, 3, torch.qint8)
        torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)
