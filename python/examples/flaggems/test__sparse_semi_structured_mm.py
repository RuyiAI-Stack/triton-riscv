import importlib

import pytest
import torch

from ._sparse_semi_structured_mm import _sparse_semi_structured_mm

# oracle: formula per upstream _sparse_semi_structured_mm.py (aten op has no
# CPU kernel). The kernel docstring (lines 65-76) fixes the semantics: for
# each group of 4 along K, meta=True keeps positions 0,1 of mat1 and drops
# 2,3; meta=False keeps 2,3 and drops 0,1 — the same math the upstream
# _sparse_semi_structured_mm_ref helper implements.


def _apply_2_4_mask(mat1, meta_bool):
    M, K = mat1.shape
    group_pos = torch.arange(K) % 4
    k_of = torch.arange(K) // 4
    meta_expanded = meta_bool[:, k_of]
    keep_mask = (group_pos[None, :] < 2) == meta_expanded
    return torch.where(keep_mask, mat1, torch.zeros_like(mat1))


def _reference(mat1, mat1_meta, mat2):
    masked = _apply_2_4_mask(mat1.float(), mat1_meta)
    return masked @ mat2.float()


@pytest.mark.parametrize("M, K4, N", [(32, 16, 24), (17, 8, 33), (64, 32, 16)])
def test_sparse_semi_structured_mm_matches_formula(M, K4, N):
    torch.manual_seed(0)
    mat1 = torch.randn(M, 4 * K4)
    mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool)
    mat2 = torch.randn(4 * K4, N)

    out = _sparse_semi_structured_mm(mat1, mat1_meta, mat2)
    ref = _reference(mat1, mat1_meta, mat2)
    assert out.shape == (M, N)
    assert out.dtype == mat1.dtype
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_sparse_semi_structured_mm_all_true_and_all_false_meta():
    torch.manual_seed(1)
    M, K4, N = 24, 8, 16
    mat1 = torch.randn(M, 4 * K4)
    mat2 = torch.randn(4 * K4, N)

    all_true = _sparse_semi_structured_mm(
        mat1, torch.ones(M, K4, dtype=torch.bool), mat2
    )
    ref_true = mat1.float().view(M, K4, 4)[:, :, :2].reshape(
        M, 2 * K4
    ) @ mat2.float().view(K4, 4, N)[:, :2].reshape(2 * K4, N)
    torch.testing.assert_close(all_true, ref_true, rtol=1e-4, atol=1e-4)

    all_false = _sparse_semi_structured_mm(
        mat1, torch.zeros(M, K4, dtype=torch.bool), mat2
    )
    ref_false = mat1.float().view(M, K4, 4)[:, :, 2:].reshape(
        M, 2 * K4
    ) @ mat2.float().view(K4, 4, N)[:, 2:].reshape(2 * K4, N)
    torch.testing.assert_close(all_false, ref_false, rtol=1e-4, atol=1e-4)


def test_sparse_semi_structured_mm_dropped_values_do_not_contribute():
    torch.manual_seed(2)
    M, K4, N = 8, 4, 8
    mat2 = torch.randn(4 * K4, N)
    base = torch.randn(M, 4 * K4)
    meta = torch.randint(0, 2, (M, K4), dtype=torch.bool)
    poisoned = base.clone()
    # Overwrite the dropped slots with huge values: they must be ignored.
    pos = torch.arange(4 * K4) % 4
    drop = (pos[None, :] < 2) != meta[:, torch.arange(4 * K4) // 4]
    poisoned[drop] = 1e6
    out = _sparse_semi_structured_mm(poisoned, meta, mat2)
    ref = _reference(base, meta, mat2)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_sparse_semi_structured_mm_float16_tolerance():
    torch.manual_seed(3)
    M, K4, N = 32, 16, 24
    mat1 = torch.randn(M, 4 * K4, dtype=torch.float16)
    mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool)
    mat2 = torch.randn(4 * K4, N, dtype=torch.float16)

    out = _sparse_semi_structured_mm(mat1, mat1_meta, mat2)
    ref = _reference(mat1, mat1_meta, mat2).to(torch.float16)
    # The kernel multiplies/sums at fp16 input precision (no tl.dot), so the
    # upstream test uses atol=0.1 for half precision; keep that budget.
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=0.1)


def test_sparse_semi_structured_mm_out_dtype_override():
    torch.manual_seed(4)
    M, K4, N = 16, 8, 24
    mat1 = torch.randn(M, 4 * K4, dtype=torch.float16)
    mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool)
    mat2 = torch.randn(4 * K4, N, dtype=torch.float16)

    out = _sparse_semi_structured_mm(mat1, mat1_meta, mat2, out_dtype=torch.float32)
    assert out.dtype == torch.float32
    ref = _reference(mat1, mat1_meta, mat2)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=0.1)


def test_sparse_semi_structured_mm_minimum_k():
    # K4=1: a single 4-wide group still selects 2 lanes.
    torch.manual_seed(5)
    mat1 = torch.randn(4, 4)
    mat1_meta = torch.tensor([[True], [False], [True], [False]])
    mat2 = torch.randn(4, 2)
    out = _sparse_semi_structured_mm(mat1, mat1_meta, mat2)
    ref = _reference(mat1, mat1_meta, mat2)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_sparse_semi_structured_mm_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._sparse_semi_structured_mm")
    launches = []
    kernel = mod.sparse_semi_structured_mm_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    mat1 = torch.randn(32, 64)
    mat1_meta = torch.randint(0, 2, (32, 16), dtype=torch.bool)
    mat2 = torch.randn(64, 24)
    out = _sparse_semi_structured_mm(mat1, mat1_meta, mat2)
    ref = _reference(mat1, mat1_meta, mat2)
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
