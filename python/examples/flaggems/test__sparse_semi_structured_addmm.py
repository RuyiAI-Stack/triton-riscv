import importlib

import pytest
import torch

from ._sparse_semi_structured_addmm import _sparse_semi_structured_addmm

# oracle: formula per upstream _sparse_semi_structured_addmm.py (aten op has
# no CPU kernel): output = alpha * (2:4-selected mat1 @ mat2) + beta * input,
# accumulated in float (the upstream test's _pytorch_ref composition).


def _apply_2_4_mask(mat1, meta_bool):
    M, K = mat1.shape
    group_pos = torch.arange(K) % 4
    k_of = torch.arange(K) // 4
    meta_expanded = meta_bool[:, k_of]
    keep_mask = (group_pos[None, :] < 2) == meta_expanded
    return torch.where(keep_mask, mat1, torch.zeros_like(mat1))


def _reference(input_tensor, mat1, meta, mat2, alpha=1.0, beta=1.0):
    mat1_sparse = _apply_2_4_mask(mat1.double(), meta)
    out = alpha * (mat1_sparse @ mat2.double()) + beta * input_tensor.double()
    return out.to(mat1.dtype)


@pytest.mark.parametrize("M, K4, N", [(32, 16, 24), (17, 8, 33)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sparse_semi_structured_addmm_matches_formula(M, K4, N, dtype):
    torch.manual_seed(0)
    input_tensor = torch.randn(M, N, dtype=dtype)
    mat1 = torch.randn(M, 4 * K4, dtype=dtype)
    mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool)
    mat2 = torch.randn(4 * K4, N, dtype=dtype)

    out = _sparse_semi_structured_addmm(input_tensor, mat1, mat1_meta, mat2)
    ref = _reference(input_tensor, mat1, mat1_meta, mat2)
    assert out.shape == (M, N) and out.dtype == dtype
    if dtype == torch.float32:
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=2e-2)
    else:
        # The kernel broadcasts fp16 products element-wise before the fp32
        # sum (no tl.dot); the upstream test budget for fp16 is atol=0.08.
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=0.08)


def test_sparse_semi_structured_addmm_alpha_beta():
    torch.manual_seed(1)
    M, K4, N = 24, 16, 32
    input_tensor = torch.randn(M, N)
    mat1 = torch.randn(M, 4 * K4)
    mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool)
    mat2 = torch.randn(4 * K4, N)

    out = _sparse_semi_structured_addmm(
        input_tensor, mat1, mat1_meta, mat2, alpha=2.5, beta=0.5
    )
    ref = _reference(input_tensor, mat1, mat1_meta, mat2, alpha=2.5, beta=0.5)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=2e-2)


def test_sparse_semi_structured_addmm_zero_beta_is_pure_input():
    torch.manual_seed(2)
    M, K4, N = 16, 8, 12
    input_tensor = torch.randn(M, N)
    mat1 = torch.randn(M, 4 * K4)
    mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool)
    mat2 = torch.randn(4 * K4, N)

    out = _sparse_semi_structured_addmm(
        input_tensor, mat1, mat1_meta, mat2, alpha=0.0, beta=-1.0
    )
    torch.testing.assert_close(out, -input_tensor, rtol=1e-5, atol=1e-5)


def test_sparse_semi_structured_addmm_out_dtype():
    torch.manual_seed(3)
    M, K4, N = 16, 8, 12
    input_tensor = torch.randn(M, N, dtype=torch.float16)
    mat1 = torch.randn(M, 4 * K4, dtype=torch.float16)
    mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool)
    mat2 = torch.randn(4 * K4, N, dtype=torch.float16)

    out = _sparse_semi_structured_addmm(
        input_tensor, mat1, mat1_meta, mat2, out_dtype=torch.float32
    )
    assert out.dtype == torch.float32
    ref = _reference(input_tensor, mat1, mat1_meta, mat2).to(torch.float32)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=0.08)


def test_sparse_semi_structured_addmm_shape_mismatch_raises():
    torch.manual_seed(4)
    M, K4, N = 8, 8, 12
    input_tensor = torch.randn(M, N)
    mat1 = torch.randn(M, 4 * K4)
    mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool)
    mat2 = torch.randn(4 * K4, N)
    with pytest.raises(AssertionError):
        _sparse_semi_structured_addmm(torch.randn(M + 1, N), mat1, mat1_meta, mat2)
    with pytest.raises(AssertionError):
        _sparse_semi_structured_addmm(input_tensor, mat1[:, :-1], mat1_meta, mat2)
    with pytest.raises(AssertionError):
        _sparse_semi_structured_addmm(input_tensor, mat1, mat1_meta[:, :-1], mat2)
    with pytest.raises(AssertionError):
        _sparse_semi_structured_addmm(
            input_tensor, mat1, mat1_meta, torch.randn(4 * K4 + 2, N)
        )


def test_sparse_semi_structured_addmm_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._sparse_semi_structured_addmm")
    launches = []
    kernel = mod.sparse_semi_structured_addmm_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    M, K4, N = 32, 16, 24
    input_tensor = torch.randn(M, N)
    mat1 = torch.randn(M, 4 * K4)
    mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool)
    mat2 = torch.randn(4 * K4, N)
    out = _sparse_semi_structured_addmm(input_tensor, mat1, mat1_meta, mat2)
    ref = _reference(input_tensor, mat1, mat1_meta, mat2)
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=2e-2)
