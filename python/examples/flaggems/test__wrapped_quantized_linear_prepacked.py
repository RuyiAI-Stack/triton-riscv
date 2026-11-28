import importlib

import pytest
import torch

from ._wrapped_quantized_linear_prepacked import _wrapped_quantized_linear_prepacked

_PACK_MAGIC = 260821.0
_PACK_VERSION = 1.0


def _make_qparams():
    return (
        torch.tensor(0.05),
        torch.tensor(127),
        torch.tensor(0.03125),
        torch.tensor(-3),
        torch.tensor(0.08),
        torch.tensor(121),
    )


def _pack_weight(weight, bias, weight_scale, weight_zero_point, n, k):
    """Build the FlagGems linear packing: 16-byte fp32 metadata header
    [scale, zero_point, magic, version], then the fp32 bias, then the int8
    quantized weight rows."""
    quantized = (torch.round(weight / weight_scale) + float(weight_zero_point)).clamp(
        -128, 127
    )
    packed = torch.empty(16 + 4 * n + n * k, dtype=torch.uint8)
    packed[:16].view(torch.float32).copy_(
        torch.tensor(
            [
                float(weight_scale),
                float(weight_zero_point),
                _PACK_MAGIC,
                _PACK_VERSION,
            ],
            dtype=torch.float32,
        )
    )
    packed[16 : 16 + 4 * n].view(torch.float32).copy_(bias)
    packed[16 + 4 * n :].view(torch.int8).copy_(quantized.to(torch.int8).reshape(-1))
    return packed


def _formula_reference(inp, weight, bias, in_s, in_zp, w_s, w_zp, out_s, out_zp, n):
    # Same arithmetic the kernel implements, composed from the packing the
    # test builds: uint8-style activation quantize, int8 weight dequant,
    # fp32 accumulate, requantized output.
    k = weight.shape[1]
    packed = _pack_weight(weight, bias, w_s, w_zp, n, k)
    bias_f = packed[16 : 16 + 4 * n].view(torch.float32)
    w_int = packed[16 + 4 * n :].view(torch.int8).reshape(n, k).float()
    q_in = (torch.round(inp.float() / float(in_s)) + float(in_zp)).clamp(0, 255)
    acc = (q_in - float(in_zp)) @ (w_int - float(w_zp)).T
    real = acc * float(in_s) * float(w_s) + bias_f
    q_out = (torch.round(real / float(out_s)) + float(out_zp)).clamp(0, 255)
    return (q_out - float(out_zp)) * float(out_s)


def _aten_reference(inp, weight, bias, in_s, in_zp, w_s, w_zp, out_s, out_zp, n):
    try:
        packed = torch.ops.aten._wrapped_linear_prepack(weight, w_s, w_zp, bias)
        return torch.ops.aten._wrapped_quantized_linear_prepacked(
            inp, in_s, in_zp, packed, out_s, out_zp, n
        )
    except RuntimeError as exc:
        # FBGEMM is not compiled into every torch build (e.g. riscv64 CI);
        # the formula composition above reproduces the reference exactly.
        if "FBGEMM" not in str(exc):
            raise
        return _formula_reference(
            inp, weight, bias, in_s, in_zp, w_s, w_zp, out_s, out_zp, n
        )


def _run(inp, weight, bias, n, noncontiguous=False):
    in_s, in_zp, w_s, w_zp, out_s, out_zp = _make_qparams()
    k = weight.shape[1]
    if noncontiguous:
        inp = (torch.randint(-8, 9, (*inp.shape[:-1], k * 2)).float() * float(in_s))[
            ..., ::2
        ]
        weight = (torch.randint(-8, 9, (n, k * 2)).float() * float(w_s))[..., ::2]
        bias = torch.zeros(n * 2)[::2]
    packed = _pack_weight(weight, bias, w_s, w_zp, n, k)
    actual = _wrapped_quantized_linear_prepacked(
        inp, in_s, in_zp, packed, out_s, out_zp, n
    )
    ref = _aten_reference(inp, weight, bias, in_s, in_zp, w_s, w_zp, out_s, out_zp, n)
    return actual, ref, float(out_s)


def test_wrapped_quantized_linear_matches_aten():
    torch.manual_seed(0)
    for leading_shape, n, k in (((2,), 4, 8), ((3,), 33, 65), ((2, 5), 16, 32)):
        inp = torch.randint(-8, 9, (*leading_shape, k)).float() * 0.05
        weight = torch.randn(n, k)
        bias = torch.randn(n)
        actual, ref, out_scale = _run(inp, weight, bias, n)
        assert actual.shape == (*leading_shape, n)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, ref, rtol=0, atol=out_scale * 1.01)


def test_wrapped_quantized_linear_non_contiguous():
    torch.manual_seed(1)
    n, k = 4, 16
    inp = torch.randint(-8, 9, (2, k)).float() * 0.05
    weight = torch.randn(n, k)
    bias = torch.randn(n)
    actual, ref, out_scale = _run(inp, weight, bias, n, noncontiguous=True)
    assert actual.shape == (2, n)
    torch.testing.assert_close(actual, ref, rtol=0, atol=out_scale * 1.01)


def test_wrapped_quantized_linear_saturation():
    torch.manual_seed(2)
    n, k = 4, 8
    # activations far outside the representable range saturate at 0 / 255
    inp = torch.randint(-320, 321, (3, k)).float() * 0.05
    weight = torch.randn(n, k)
    bias = torch.randn(n)
    actual, ref, out_scale = _run(inp, weight, bias, n)
    torch.testing.assert_close(actual, ref, rtol=0, atol=out_scale * 1.01)


def test_wrapped_quantized_linear_empty_batch():
    n, k = 5, 7
    inp = torch.empty(2, 0, k)
    in_s, in_zp, w_s, w_zp, out_s, out_zp = _make_qparams()
    packed = _pack_weight(torch.randn(n, k), torch.randn(n), w_s, w_zp, n, k)
    out = _wrapped_quantized_linear_prepacked(
        inp, in_s, in_zp, packed, out_s, out_zp, n
    )
    assert out.shape == (2, 0, n)
    assert out.numel() == 0


def test_wrapped_quantized_linear_zero_out_channel():
    torch.manual_seed(3)
    n, k = 0, 7
    inp = torch.randn(2, k)
    in_s, in_zp, w_s, w_zp, out_s, out_zp = _make_qparams()
    packed = _pack_weight(torch.empty(0, k), torch.empty(0), w_s, w_zp, n, k)
    out = _wrapped_quantized_linear_prepacked(
        inp, in_s, in_zp, packed, out_s, out_zp, n
    )
    assert out.shape == (2, 0)


def test_wrapped_quantized_linear_zero_in_features():
    # K == 0: the output is the dequantized uint8 zero, deterministically
    n = 5
    inp = torch.randn(2, 0)
    in_s, in_zp, w_s, w_zp, out_s, out_zp = _make_qparams()
    packed = _pack_weight(torch.empty(n, 0), torch.randn(n), w_s, w_zp, n, 0)
    out = _wrapped_quantized_linear_prepacked(
        inp, in_s, in_zp, packed, out_s, out_zp, n
    )
    assert out.shape == (2, n)
    expected = -float(out_zp) * float(out_s)
    torch.testing.assert_close(out, torch.full((2, n), expected), rtol=0, atol=1e-6)


def test_wrapped_quantized_linear_errors():
    torch.manual_seed(4)
    n, k = 5, 7
    in_s, in_zp, w_s, w_zp, out_s, out_zp = _make_qparams()
    weight = torch.randn(n, k)
    bias = torch.randn(n)
    packed = _pack_weight(weight, bias, w_s, w_zp, n, k)
    with pytest.raises(RuntimeError, match="larger than or equal to 2"):
        _wrapped_quantized_linear_prepacked(
            torch.randn(k), in_s, in_zp, packed, out_s, out_zp, n
        )
    with pytest.raises(RuntimeError, match="Quantize only works on Float Tensor"):
        _wrapped_quantized_linear_prepacked(
            torch.randn(2, k, dtype=torch.float64),
            in_s,
            in_zp,
            packed,
            out_s,
            out_zp,
            n,
        )
    with pytest.raises(RuntimeError, match="out_channel must be non-negative"):
        _wrapped_quantized_linear_prepacked(
            torch.randn(2, k), in_s, in_zp, packed, out_s, out_zp, -1
        )
    with pytest.raises(RuntimeError, match="input_scale must contain one element"):
        _wrapped_quantized_linear_prepacked(
            torch.randn(2, k),
            torch.tensor([0.05, 0.1]),
            in_zp,
            packed,
            out_s,
            out_zp,
            n,
        )


def test_wrapped_quantized_linear_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._wrapped_quantized_linear_prepacked")
    launches = []
    kernel = mod._wrapped_quantized_linear_prepacked_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(5)
    n, k = 4, 8
    inp = torch.randint(-8, 9, (2, k)).float() * 0.05
    weight = torch.randn(n, k)
    bias = torch.randn(n)
    actual, ref, out_scale = _run(inp, weight, bias, n)
    assert len(launches) >= 1
    torch.testing.assert_close(actual, ref, rtol=0, atol=out_scale * 1.01)
