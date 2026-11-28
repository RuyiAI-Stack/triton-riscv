import importlib

import pytest
import torch

from ._weight_int4pack_mm_with_scales_and_zeros import (
    # atol 1e-2 is tighter than the upstream suite budget for this op
    # (tests/test_weight_int4pack_mm_with_scales_and_zeros.py uses atol=0.1).
    _weight_int4pack_mm_with_scales_and_zeros,
)

# oracle: formula per upstream _weight_int4pack_mm_with_scales_and_zeros.py
# (no aten CPU kernel): mat2 (N, K//2) uint8 packs two unsigned int4 values
# per byte (low nibble = even column k=2j, high = odd k=2j+1, conventions
# block lines 40-62); dequant w[k, n] = (q - qZeros[k // gs, n]) *
# qScale[k // gs, n]; the output is A @ W_deq with W_deq shaped (K, N) — NOT
# transposed (docstring line 59) — rounded to A's dtype before the dot
# (kernel lines 149-166).


def _pack(w_kn):
    # w_kn: (K, N) int32 raw int4 values 0..15 -> (N, K//2) byte-pair packed.
    even = (w_kn[0::2] & 0xF).to(torch.uint8)
    odd = (w_kn[1::2] & 0xF).to(torch.uint8)
    mat2 = ((odd << 4) | even).t()
    return mat2.contiguous()


def _reference(A, mat2, q_group_size, q_scale, q_zeros):
    N, K2 = mat2.shape
    K = K2 * 2
    low = (mat2.to(torch.int32) & 0xF).float()
    high = ((mat2.to(torch.int32) >> 4) & 0xF).float()
    w = torch.empty((K, N), dtype=torch.float32)
    # Per-row group index g = k // q_group_size expands the (G, N) tables to
    # (K, N), so w[k, n] uses the group of row k (docstring formula).
    g = torch.arange(K) // q_group_size
    zero_kn = q_zeros.float()[g]
    scale_kn = q_scale.float()[g]
    w[0::2] = (low.T - zero_kn[0::2]) * scale_kn[0::2]
    w[1::2] = (high.T - zero_kn[1::2]) * scale_kn[1::2]
    # The kernel casts the dequantized weight to A's dtype before the dot.
    return (A.float() @ w.to(A.dtype).float()).to(A.dtype)


def _make_case(M, K, N, q_group_size, dtype):
    A = torch.randn(M, K, dtype=dtype) * 0.1
    w_int4 = torch.randint(0, 16, (K, N), dtype=torch.int32)
    G = K // q_group_size
    q_scale = torch.randn(G, N).abs() + 0.01
    q_zeros = torch.randn(G, N) * 2.0
    return (
        A,
        _pack(w_int4),
        q_group_size,
        q_scale.to(dtype),
        q_zeros.to(dtype),
    )


@pytest.mark.parametrize(
    "M, K, N, q_group_size",
    [(4, 16, 8, 4), (8, 32, 16, 8), (4, 64, 16, 16), (8, 128, 32, 2)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_weight_int4pack_mm_with_scales_and_zeros_matches_formula(
    M, K, N, q_group_size, dtype
):
    torch.manual_seed(0)
    A, mat2, gs, q_scale, q_zeros = _make_case(M, K, N, q_group_size, dtype)
    out = _weight_int4pack_mm_with_scales_and_zeros(A, mat2, gs, q_scale, q_zeros)
    ref = _reference(A, mat2, gs, q_scale, q_zeros)
    assert out.shape == (M, N) and out.dtype == A.dtype
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_weight_int4pack_mm_with_scales_and_zeros_bfloat16():
    torch.manual_seed(1)
    A, mat2, gs, q_scale, q_zeros = _make_case(8, 32, 16, 8, torch.bfloat16)
    out = _weight_int4pack_mm_with_scales_and_zeros(A, mat2, gs, q_scale, q_zeros)
    ref = _reference(A, mat2, gs, q_scale, q_zeros)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_weight_int4pack_mm_with_scales_and_zeros_full_width_group():
    # qGroupSize == K: a single scale/zero row serves every column.
    torch.manual_seed(2)
    A, mat2, gs, q_scale, q_zeros = _make_case(4, 32, 8, 32, torch.float32)
    out = _weight_int4pack_mm_with_scales_and_zeros(A, mat2, gs, q_scale, q_zeros)
    ref = _reference(A, mat2, gs, q_scale, q_zeros)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_weight_int4pack_mm_with_scales_and_zeros_zero_scale():
    torch.manual_seed(3)
    A = torch.randn(4, 16) * 0.1
    w_int4 = torch.randint(0, 16, (16, 8), dtype=torch.int32)
    q_scale = torch.zeros(1, 8)
    q_zeros = torch.zeros(1, 8)
    out = _weight_int4pack_mm_with_scales_and_zeros(
        A, _pack(w_int4), 16, q_scale, q_zeros
    )
    assert torch.all(out == 0)


def test_weight_int4pack_mm_with_scales_and_zeros_errors_are_assertions():
    torch.manual_seed(4)
    A = torch.randn(8, 32)
    mat2 = _pack(torch.randint(0, 16, (32, 16), dtype=torch.int32))
    q_scale = torch.rand(2, 16)
    q_zeros = torch.rand(2, 16)
    with pytest.raises(AssertionError):
        _weight_int4pack_mm_with_scales_and_zeros(A, mat2[:, :8], 16, q_scale, q_zeros)
    with pytest.raises(AssertionError):
        _weight_int4pack_mm_with_scales_and_zeros(A, mat2, 12, q_scale, q_zeros)
    with pytest.raises(AssertionError):
        _weight_int4pack_mm_with_scales_and_zeros(
            A, mat2, 16, torch.rand(3, 16), q_zeros
        )
    with pytest.raises(AssertionError):
        _weight_int4pack_mm_with_scales_and_zeros(
            A, mat2, 16, q_scale, torch.rand(3, 16)
        )
    with pytest.raises(AssertionError):
        _weight_int4pack_mm_with_scales_and_zeros(A, mat2.int(), 16, q_scale, q_zeros)
    with pytest.raises(AssertionError):
        _weight_int4pack_mm_with_scales_and_zeros(
            A.int(), mat2, 16, q_scale.int(), q_zeros.int()
        )


def test_weight_int4pack_mm_with_scales_and_zeros_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._weight_int4pack_mm_with_scales_and_zeros")
    launches = []
    kernel = mod._weight_int4pack_mm_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    A, mat2, gs, q_scale, q_zeros = _make_case(8, 32, 16, 8, torch.float32)
    out = _weight_int4pack_mm_with_scales_and_zeros(A, mat2, gs, q_scale, q_zeros)
    ref = _reference(A, mat2, gs, q_scale, q_zeros)
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
