import importlib

import pytest
import torch

from .weight_int4pack_mm import weight_int4pack_mm

# atol 1e-2 matches the upstream suite budget for this op
# (tests/test_weight_int4pack_mm.py compares with atol=1e-2).

# oracle: formula per upstream weight_int4pack_mm.py (aten._weight_int4pack_mm
# has no CPU kernel and ATen's own packer uses the opposite nibble order):
# each byte of mat2 (N, K//2) packs two unsigned int4 values — low nibble =
# even column 2j, high nibble = odd column 2j+1 (docstring lines 21-33) — and
# dequant is w = (q - zero) * scale with per-group qScaleAndZeros[g, n, 0]
# scale / [1] zero, g = k // qGroupSize (kernel lines 133-149). The output is
# A @ dequant(W).T computed in fp32.


def _pack(w_int4):
    n, k = w_int4.shape
    even = (w_int4[:, 0::2] & 0xF).to(torch.uint8)
    odd = (w_int4[:, 1::2] & 0xF).to(torch.uint8)
    return ((odd << 4) | even).contiguous()


def _ref(A, mat2, q_group_size, scale, zero):
    n, k2 = mat2.shape
    K = k2 * 2
    low = (mat2.to(torch.int32) & 0xF).float()
    high = ((mat2.to(torch.int32) >> 4) & 0xF).float()
    # Per-column group index g = k // qGroupSize expands the (G, N) scale/zero
    # tables to (K, N) before the transpose, so w[n, k] uses g of column k.
    g = torch.arange(K) // q_group_size
    zero_kn = zero.float()[g].t()
    scale_kn = scale.float()[g].t()
    w = torch.empty((n, K), dtype=torch.float32)
    w[:, 0::2] = (low - zero_kn[:, 0::2]) * scale_kn[:, 0::2]
    w[:, 1::2] = (high - zero_kn[:, 1::2]) * scale_kn[:, 1::2]
    return (A.float() @ w.T).to(A.dtype)


def _make_case(M, N, K, q_group_size, dtype):
    A = torch.randn(M, K, dtype=dtype) * 0.25
    w_int4 = torch.randint(0, 16, (N, K), dtype=torch.int32)
    num_groups = K // q_group_size
    scale = torch.rand(num_groups, N) * 1.5 + 0.5
    zero = torch.randint(4, 11, (num_groups, N)).float()
    q_sz = torch.stack([scale, zero], dim=-1).to(dtype)
    return A, _pack(w_int4), q_sz, scale, zero


@pytest.mark.parametrize(
    "M, N, K, q_group_size",
    [(8, 16, 64, 32), (16, 16, 128, 32), (16, 32, 128, 64), (8, 16, 64, 64)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_weight_int4pack_mm_matches_formula(M, N, K, q_group_size, dtype):
    torch.manual_seed(0)
    A, mat2, q_sz, scale, zero = _make_case(M, N, K, q_group_size, dtype)
    out = weight_int4pack_mm(A, mat2, q_group_size, q_sz)
    ref = _ref(A, mat2, q_group_size, scale, zero)
    assert out.shape == (M, N) and out.dtype == A.dtype
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_weight_int4pack_mm_bfloat16():
    torch.manual_seed(1)
    A, mat2, q_sz, scale, zero = _make_case(8, 16, 64, 32, torch.bfloat16)
    out = weight_int4pack_mm(A, mat2, 32, q_sz)
    ref = _ref(A, mat2, 32, scale, zero)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_weight_int4pack_mm_zero_scale_yields_zeros():
    torch.manual_seed(2)
    A = torch.randn(4, 32)
    w_int4 = torch.randint(0, 16, (8, 32), dtype=torch.int32)
    scale = torch.zeros(1, 8)
    zero = torch.full((1, 8), 8.0)
    q_sz = torch.stack([scale, zero], dim=-1)
    out = weight_int4pack_mm(A, _pack(w_int4), 32, q_sz)
    assert torch.all(out == 0)


def test_weight_int4pack_mm_single_output_channel():
    torch.manual_seed(3)
    A, mat2, q_sz, scale, zero = _make_case(4, 1, 32, 32, torch.float32)
    out = weight_int4pack_mm(A, mat2, 32, q_sz)
    ref = _ref(A, mat2, 32, scale, zero)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_weight_int4pack_mm_non_contiguous_activation():
    # The wrapper makes A contiguous; a row-sliced view must still work.
    torch.manual_seed(4)
    base = torch.randn(16, 64) * 0.25
    A = base[::2]  # (8, 64) row-sliced view, stride (128, 1) -> non-contiguous
    assert not A.is_contiguous()
    w_int4 = torch.randint(0, 16, (8, 64), dtype=torch.int32)
    scale = torch.rand(2, 8) * 1.5 + 0.5
    zero = torch.randint(4, 11, (2, 8)).float()
    q_sz = torch.stack([scale, zero], dim=-1)
    out = weight_int4pack_mm(A, _pack(w_int4), 32, q_sz)
    ref = _ref(A.contiguous(), _pack(w_int4), 32, scale, zero)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_weight_int4pack_mm_errors_are_assertions():
    torch.manual_seed(5)
    A = torch.randn(8, 64)
    mat2 = _pack(torch.randint(0, 16, (16, 64), dtype=torch.int32))
    q_sz = torch.stack(
        [torch.rand(2, 16) + 0.5, torch.randint(4, 11, (2, 16)).float()], dim=-1
    )
    with pytest.raises(AssertionError):
        weight_int4pack_mm(A, mat2[:, :16], 32, q_sz)  # K no longer 2 * K_half
    with pytest.raises(AssertionError):
        weight_int4pack_mm(A, mat2, 48, q_sz)  # K % qGroupSize != 0
    with pytest.raises(AssertionError):
        weight_int4pack_mm(A, mat2, 33, q_sz)  # qGroupSize must be even
    with pytest.raises(AssertionError):
        weight_int4pack_mm(A, mat2.int(), 32, q_sz)  # mat2 must be uint8
    with pytest.raises(AssertionError):
        weight_int4pack_mm(A.int(), mat2, 32, q_sz)  # A must be floating point


def test_weight_int4pack_mm_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.weight_int4pack_mm")
    launches = []
    kernel = mod.weight_int4pack_mm_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    A, mat2, q_sz, scale, zero = _make_case(8, 16, 64, 32, torch.float32)
    out = weight_int4pack_mm(A, mat2, 32, q_sz)
    ref = _ref(A, mat2, 32, scale, zero)
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
