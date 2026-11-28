import importlib

import pytest
import torch

from ._dyn_quant_matmul_4bit import _dyn_quant_matmul_4bit


def _pack(w, scales, bias):
    """Portable fp32 ABI: N*K/2 byte values, then N*(K/group) row-major
    scales, then optionally N bias values."""
    parts = [w.reshape(-1).float(), scales.reshape(-1).float()]
    if bias is not None:
        parts.append(bias.float())
    return torch.cat(parts)


def _reference(x, weights, scales, bias, group):
    """Independent CPU quantization and integer dot (from the upstream suite)."""
    x = x.float()
    lo = x.amin(1).clamp(max=0)
    hi = x.amax(1).clamp(min=0)
    mult = torch.where(lo == hi, 1.0, torch.div(torch.full_like(lo, 255.0), hi - lo))
    scale = torch.where(mult != 0, torch.div(torch.ones_like(mult), mult), 0.0)
    a, b = lo * mult, hi * mult
    zp = torch.where((-128.0 + a) + (127.0 + b) > 0, -128.0 - a, 127.0 - b)
    zp = zp.clamp(-128, 127).round().to(torch.int32)
    values = x * mult[:, None]
    rounded = values.sign() * (values.abs() + 0.5).floor()
    q = (rounded.to(torch.int32) + zp[:, None]).clamp(-128, 127) - zp[:, None]
    n, half_k = weights.shape
    w = torch.empty((n, half_k * 2), dtype=torch.int32)
    w[:, 0::2] = (weights.to(torch.int32) & 15) - 8
    w[:, 1::2] = (weights.to(torch.int32) >> 4) - 8
    result = torch.zeros((x.shape[0], n), dtype=torch.float32)
    for g in range(x.shape[1] // group):
        start = g * group
        dot = q[:, start : start + group].to(torch.int64) @ w[
            :, start : start + group
        ].T.to(torch.int64)
        result += dot.to(torch.int32).float() * scales[:, g].float()
    result *= scale[:, None]
    if bias is not None:
        result += bias.float()
    return result


def _case(m, n, k, group, dtype, bias, noncontiguous=False):
    gen = torch.Generator().manual_seed(43)
    x = torch.randn((m, k), generator=gen).to(dtype)
    w = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8)
    scales = torch.rand((n, k // group), generator=gen) + 0.1
    b = torch.randn(n, generator=gen) if bias else None
    packed = _pack(w, scales, b)
    if m == 0 or n == 0:
        expected = torch.empty((m, n), dtype=dtype)
    elif dtype == torch.float32:
        # aten consumes the same portable fp32 ABI bit-exactly for fp32 input;
        # the formula reference covers bf16 (aten needs the fp32 path).
        expected = torch.ops.aten._dyn_quant_matmul_4bit(
            x.float(), packed, group, k, n
        ).to(dtype)
    else:
        expected = _reference(x, w, scales, b, group).to(dtype)
    xd = x
    if noncontiguous:
        storage = torch.empty((k, m * 2), dtype=dtype)
        xd = storage[:, ::2].T
        xd.copy_(x)
    actual = _dyn_quant_matmul_4bit(xd, packed, group, k, n)
    assert actual.dtype == dtype
    assert actual.shape == (m, n)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5 if dtype == torch.float32 else 0.008,
        atol=2e-5 if dtype == torch.float32 else 0.008,
    )
    if dtype == torch.float32 and m and n:
        torch.testing.assert_close(
            actual, _reference(x, w, scales, b, group), rtol=1e-5, atol=2e-5
        )


def test_dyn_quant_matmul_4bit_channelwise():
    for dtype in (torch.float32, torch.bfloat16):
        for bias in (False, True):
            for m, n, k in ((1, 1, 2), (7, 31, 64), (4, 16, 130)):
                _case(m, n, k, k, dtype, bias)


def test_dyn_quant_matmul_4bit_groupwise():
    for bias in (False, True):
        for m, n, k, group in ((1, 1, 32, 32), (7, 31, 64, 32), (5, 12, 96, 32)):
            _case(m, n, k, group, torch.float32, bias)


def test_dyn_quant_matmul_4bit_empty_inputs():
    _case(0, 31, 64, 64, torch.float32, True)
    _case(1, 0, 64, 64, torch.float32, False)


def test_dyn_quant_matmul_4bit_noncontiguous():
    for dtype in (torch.float32, torch.bfloat16):
        _case(7, 31, 66, 66, dtype, True, noncontiguous=True)


def test_dyn_quant_matmul_4bit_rounding_and_nibbles():
    # half-integers distinguish away-from-zero activation rounding from
    # nearest-even zero-point rounding; nibble order is pinned exactly
    x = torch.tensor(
        [
            [-128, 127, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [3] * 8,
            [-3] * 8,
            [-127.5, 127.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
            [3e38, -3e38, 0, 1, -1, 2, -2, 0],
        ],
        dtype=torch.float32,
    )
    w = torch.tensor([[0xF0, 0x0F, 0x81, 0x18], [0xFF] * 4, [0] * 4], dtype=torch.uint8)
    s = torch.tensor([[1.0], [0.5], [2.0]])
    ref = _reference(x, w, s, None, 8)
    actual = _dyn_quant_matmul_4bit(x, _pack(w, s, None), 8, 8, 3)
    torch.testing.assert_close(actual, ref, rtol=0, atol=0)


def test_dyn_quant_matmul_4bit_invalid():
    x = torch.empty(2, 64)
    p = torch.empty(66)
    block, k, n = 64, 64, 2
    cases = {
        "rank": (x.unsqueeze(0), p, block, k, n, "two-dimensional"),
        "dtype": (x.to(torch.float16), p, block, k, n, "float32 or bfloat16"),
        "zero_k": (x, p, block, 0, n, "positive and even"),
        "odd_k": (x, p, block, 63, n, "positive and even"),
        "negative_n": (x, p, block, k, -1, "nonnegative"),
        "mismatch_k": (x, p, block, 32, n, "inp.size"),
        "zero_block": (x, p, 0, k, n, "block_size"),
        "bad_block": (x, p, 16, k, n, "block_size"),
        "nondividing_block": (x, p, 96, k, n, "block_size"),
        "bf16_group": (x.to(torch.bfloat16), p, 32, k, n, "bfloat16 requires"),
        "opaque": (x, p.to(torch.uint8), block, k, n, "opaque uint8"),
        "packed_dtype": (x, p.to(torch.bfloat16), block, k, n, "float32 portable"),
        "packed_rank": (x, p.reshape(2, 33), block, k, n, "one-dimensional"),
        "packed_stride": (x, torch.empty(132)[::2], block, k, n, "contiguous"),
        "short": (x, p[:-1], block, k, n, "length"),
        "long": (x, torch.empty(69), block, k, n, "length"),
    }
    for fault, (xi, pi, bi, ki, ni, match) in cases.items():
        with pytest.raises(RuntimeError, match=match):
            _dyn_quant_matmul_4bit(xi, pi, bi, ki, ni)


def test_dyn_quant_matmul_4bit_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._dyn_quant_matmul_4bit")
    for kernel_name in ("_quantize_rows", "_w4a8_matmul"):
        kernel = getattr(mod, kernel_name)
        assert hasattr(kernel, "run"), f"{kernel_name} must be a triton kernel"
    launches = []
    for kernel_name in ("_quantize_rows", "_w4a8_matmul"):
        kernel = getattr(mod, kernel_name)
        real_run = kernel.run

        def spy_run(*args, _real=real_run, **kwargs):
            launches.append(kwargs.get("grid"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
    gen = torch.Generator().manual_seed(43)
    m, n, k = 7, 31, 64
    x = torch.randn((m, k), generator=gen)
    w = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8)
    scales = torch.rand((n, 1), generator=gen) + 0.1
    packed = _pack(w, scales, None)
    actual = _dyn_quant_matmul_4bit(x, packed, k, k, n)
    assert len(launches) >= 2, "both the quantize and the matmul kernel must run"
    torch.testing.assert_close(
        actual, _reference(x, w, scales, None, k), rtol=1e-5, atol=2e-5
    )
