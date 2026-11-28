import importlib

import pytest
import torch

from .make_per_tensor_quantized_tensor import (
    _make_per_tensor_quantized_tensor,
    _make_per_tensor_quantized_tensor_out,
)


def test_make_per_tensor_int_dtypes_wrap_verbatim():
    torch.manual_seed(0)
    for int_dtype, qdtype in (
        (torch.int8, torch.qint8),
        (torch.uint8, torch.quint8),
        (torch.int32, torch.qint32),
    ):
        for scale, zp in ((0.1, 5), (1e-4, 0), (32.0, -3), (0.5, 127)):
            x = torch.randint(
                -100 if int_dtype != torch.uint8 else 0,
                101,
                (3, 17),
                dtype=int_dtype,
            )
            q = _make_per_tensor_quantized_tensor(x, scale, zp)
            assert q.is_quantized
            assert q.dtype == qdtype
            assert q.q_scale() == scale
            assert q.q_zero_point() == zp
            assert q.qscheme() == torch.per_tensor_affine
            torch.testing.assert_close(q.int_repr(), x, rtol=0, atol=0)
            # fresh allocation: does not alias the input storage
            assert q.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()


def test_make_per_tensor_dequant_formula():
    x = torch.tensor([1, -2, 3, 127, -128], dtype=torch.int8)
    q = _make_per_tensor_quantized_tensor(x, 0.25, 4)
    expected = (x.to(torch.float32) - 4) * 0.25
    torch.testing.assert_close(torch.dequantize(q), expected, rtol=0, atol=0)


def test_make_per_tensor_matches_aten_and_quantize_equivalence():
    torch.manual_seed(1)
    x = torch.randint(-128, 128, (4, 6), dtype=torch.int8)
    q = _make_per_tensor_quantized_tensor(x, 0.05, 3)
    ref = torch._make_per_tensor_quantized_tensor(x, 0.05, 3)
    assert q.dtype == ref.dtype
    assert q.q_scale() == ref.q_scale() and q.q_zero_point() == ref.q_zero_point()
    torch.testing.assert_close(q.int_repr(), ref.int_repr(), rtol=0, atol=0)
    # cross-check: wrapping the int_repr of a quantize_per_tensor result with
    # the same qparams reproduces its dequantized values
    y = torch.randn(4, 6)
    qy = torch.quantize_per_tensor(y, 0.1, 7, torch.qint8)
    q2 = _make_per_tensor_quantized_tensor(qy.int_repr(), 0.1, 7)
    torch.testing.assert_close(
        torch.dequantize(q2), torch.dequantize(qy), rtol=0, atol=0
    )


def test_make_per_tensor_non_contiguous_input():
    torch.manual_seed(2)
    base = torch.arange(24, dtype=torch.int8).reshape(4, 6)
    for x in (base.t(), base[::2, ::2]):
        assert not x.is_contiguous()
        q = _make_per_tensor_quantized_tensor(x, 0.1, 3)
        torch.testing.assert_close(q.int_repr(), x, rtol=0, atol=0)
        assert q.int_repr().is_contiguous()


def test_make_per_tensor_empty_tensor():
    q = _make_per_tensor_quantized_tensor(torch.empty(3, 0, dtype=torch.int8), 0.1, 3)
    assert q.dtype == torch.qint8
    assert q.int_repr().shape == (3, 0)


def test_make_per_tensor_unsupported_dtype():
    with pytest.raises(KeyError, match="torch.float32"):
        _make_per_tensor_quantized_tensor(torch.randn(3), 0.1, 3)


def test_make_per_tensor_out_writes_in_place():
    torch.manual_seed(3)
    x = torch.randint(-128, 128, (3, 17), dtype=torch.int8)
    out = torch._empty_affine_quantized(
        (3, 17), scale=0.1, zero_point=3, dtype=torch.qint8
    )
    ret = _make_per_tensor_quantized_tensor_out(x, 0.1, 3, out=out)
    assert ret is out
    torch.testing.assert_close(out.int_repr(), x, rtol=0, atol=0)
    assert out.q_scale() == 0.1 and out.q_zero_point() == 3


def test_make_per_tensor_out_empty_is_untouched():
    out = torch._empty_affine_quantized(
        (0,), scale=0.1, zero_point=3, dtype=torch.qint8
    )
    ret = _make_per_tensor_quantized_tensor_out(
        torch.empty(0, dtype=torch.int8), 0.1, 3, out=out
    )
    assert ret is out
    assert out.int_repr().shape == (0,)


def test_make_per_tensor_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.make_per_tensor_quantized_tensor")
    contig = torch.randint(-128, 128, (3, 17), dtype=torch.int8)
    strided = torch.arange(24, dtype=torch.int8).reshape(4, 6).t()
    for kernel_name, inp in (
        ("_copy_1d_kernel", contig),
        ("_gather_1d_kernel", strided),
    ):
        launches = []
        kernel = getattr(mod, kernel_name)
        real_run = kernel.run

        def spy_run(*args, _real=real_run, **kwargs):
            launches.append(kwargs.get("grid"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
        q = _make_per_tensor_quantized_tensor(inp, 0.1, 3)
        assert len(launches) >= 1, f"{kernel_name} must be launched"
        torch.testing.assert_close(q.int_repr(), inp, rtol=0, atol=0)
