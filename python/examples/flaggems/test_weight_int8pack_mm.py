import importlib

import pytest
import torch

from .weight_int8pack_mm import weight_int8pack_mm

# Dequantized int8 weights are exact; the dot runs with reduced
# precision (allow_tf32 upstream). A probe measured the formula-vs-aten
# gap at 8.5e-4, leaving ~10x margin under atol 1e-2.

# oracle: aten._weight_int8pack_mm runs on CPU for fp32 activations (probed
# on torch 2.15.0+cpu, layout B (N, K) int8 + per-channel fp32 scales) — the
# direct oracle. fp16/bf16 activations use the upstream test's formula
# reference: (A.float() @ B.float().T) * scales, cast back to A's dtype
# (kernel computes the dot in fp32 with allow_tf32=True, hence the 1e-2
# budget for half precision).


def _int8(shape):
    return torch.randint(-128, 128, shape, dtype=torch.int8)


@pytest.mark.parametrize("M, N, K", [(16, 32, 64), (33, 17, 129)])
def test_weight_int8pack_mm_fp32_matches_aten(M, N, K):
    torch.manual_seed(0)
    A = torch.randn(M, K)
    B = _int8((N, K))
    scales = torch.rand(N) + 0.5

    out = weight_int8pack_mm(A, B, scales)
    ref = torch.ops.aten._weight_int8pack_mm(A, B, scales)
    assert out.shape == (M, N) and out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_weight_int8pack_mm_half_precision_matches_formula(dtype):
    torch.manual_seed(1)
    M, N, K = 16, 32, 64
    A = torch.randn(M, K, dtype=dtype) * 0.25
    B = _int8((N, K))
    scales = (torch.rand(N) * 0.005).to(dtype)

    out = weight_int8pack_mm(A, B, scales)
    ref = ((A.float() @ B.float().T) * scales.float()).to(dtype)
    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_weight_int8pack_mm_matches_formula_fp32():
    torch.manual_seed(2)
    M, N, K = 17, 33, 65
    A = torch.randn(M, K)
    B = _int8((N, K))
    scales = torch.rand(N) + 0.5
    out = weight_int8pack_mm(A, B, scales)
    ref = (A @ B.float().T) * scales
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_weight_int8pack_mm_zero_and_negative_scales():
    torch.manual_seed(3)
    A = torch.randn(8, 32)
    B = _int8((16, 32))
    zeros = weight_int8pack_mm(A, B, torch.zeros(16))
    assert torch.all(zeros == 0)
    pos = weight_int8pack_mm(A, B, torch.ones(16))
    neg = weight_int8pack_mm(A, B, -torch.ones(16))
    torch.testing.assert_close(neg, -pos, rtol=1e-5, atol=1e-5)


def test_weight_int8pack_mm_single_output_channel():
    torch.manual_seed(4)
    A = torch.randn(4, 32)
    B = _int8((1, 32))
    scales = torch.tensor([2.0])
    out = weight_int8pack_mm(A, B, scales)
    ref = torch.ops.aten._weight_int8pack_mm(A, B, scales)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_weight_int8pack_mm_errors_are_assertions():
    torch.manual_seed(5)
    A = torch.randn(8, 64)
    B = _int8((16, 64))
    scales = torch.ones(16)
    with pytest.raises(AssertionError):
        weight_int8pack_mm(A, _int8((16, 32)), scales)  # K mismatch
    with pytest.raises(AssertionError):
        weight_int8pack_mm(A, B, torch.ones(8))  # N mismatch
    with pytest.raises(AssertionError):
        weight_int8pack_mm(A, B.float(), scales)  # B must be int8


def test_weight_int8pack_mm_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.weight_int8pack_mm")
    launches = []
    kernel = mod.weight_int8pack_mm_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    A = torch.randn(16, 64)
    B = _int8((32, 64))
    scales = torch.rand(32) + 0.5
    out = weight_int8pack_mm(A, B, scales)
    ref = torch.ops.aten._weight_int8pack_mm(A, B, scales)
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
