import importlib

import pytest
import torch

from .sspaddmm import sspaddmm, sspaddmm_out


def test_sspaddmm_matches_torch_two_shapes():
    torch.manual_seed(0)
    for k, n in ((5, 4), (8, 2)):
        inp = torch.randn(3, n).to_sparse().coalesce()
        mat1 = torch.randn(3, k).to_sparse().coalesce()
        mat2 = torch.randn(k, n, dtype=torch.float32)
        out = sspaddmm(inp, mat1, mat2, beta=1.5, alpha=0.5)
        ref = torch.sspaddmm(inp, mat1, mat2, beta=1.5, alpha=0.5)
        assert out.layout == torch.sparse_coo
        assert out.shape == (3, n) and out.dtype == torch.float32
        out_c = out.coalesce() if not out.is_coalesced() else out
        ref_c = ref.coalesce() if not ref.is_coalesced() else ref
        assert out_c._nnz() == ref_c._nnz()
        torch.testing.assert_close(
            out_c.indices(), ref_c.indices(), rtol=0, atol=0, msg=f"k={k}"
        )
        # 1e-5: fp32 dot order differs from torch's kernel.
        torch.testing.assert_close(out_c.values(), ref_c.values(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-5, atol=1e-5)


def test_sspaddmm_default_beta_alpha_match_torch():
    torch.manual_seed(1)
    inp = torch.randn(4, 5).to_sparse().coalesce()
    mat1 = torch.randn(4, 6).to_sparse().coalesce()
    mat2 = torch.randn(6, 5, dtype=torch.float32)
    out = sspaddmm(inp, mat1, mat2)
    ref = torch.sspaddmm(inp, mat1, mat2)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-5, atol=1e-5)


def test_sspaddmm_disjoint_structures_union():
    # torch.sspaddmm's mm contribution stores EVERY (active row, column)
    # pair -- structural zeros survive coalescing (probed on 2.15.0+cpu:
    # 3x3 grid here coalesces to nnz 9, values holding explicit 0.0), so
    # the union is the row-block product pattern, not a pruned merge.
    torch.manual_seed(2)
    base = torch.tensor([[1.0, 0, 0], [0, 0, 0], [0, 2, 0]])
    inp = base.to_sparse().coalesce()
    m1base = torch.tensor([[0.0, 0, 5], [0, 3, 0], [0, 0, 1]])
    mat1 = m1base.to_sparse().coalesce()
    mat2 = torch.eye(3, dtype=torch.float32)
    out = sspaddmm(inp, mat1, mat2, beta=2.0, alpha=3.0)
    ref = torch.sspaddmm(inp, mat1, mat2, beta=2.0, alpha=3.0)
    out_c = out.coalesce() if not out.is_coalesced() else out
    ref_c = ref.coalesce() if not ref.is_coalesced() else ref
    assert out_c._nnz() == ref_c._nnz()
    torch.testing.assert_close(out_c.indices(), ref_c.indices(), rtol=0, atol=0)
    torch.testing.assert_close(out_c.values(), ref_c.values(), rtol=0, atol=0)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-5, atol=1e-5)


def test_sspaddmm_fp64_arange_exact():
    inp = (
        torch.tensor([[1.0, 0, 2], [0, 3, 0]]).to(torch.float64).to_sparse().coalesce()
    )
    mat1 = torch.arange(2 * 4, dtype=torch.float64).reshape(2, 4).to_sparse().coalesce()
    mat2 = torch.arange(4 * 3, dtype=torch.float64).reshape(4, 3) / 3
    out = sspaddmm(inp, mat1, mat2, beta=2.0, alpha=3.0)
    ref = torch.sspaddmm(inp, mat1, mat2, beta=2.0, alpha=3.0)
    out_c = out.coalesce() if not out.is_coalesced() else out
    ref_c = ref.coalesce() if not ref.is_coalesced() else ref
    torch.testing.assert_close(out_c.values(), ref_c.values(), rtol=0, atol=0)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=0, atol=0)


def test_sspaddmm_int64_small_values_exact():
    inp = torch.tensor([[1, 0, 2], [0, 3, 0]]).to_sparse().coalesce()
    mat1 = torch.arange(2 * 4, dtype=torch.int64).reshape(2, 4).to_sparse().coalesce()
    mat2 = torch.arange(4 * 3, dtype=torch.int64).reshape(4, 3) % 3
    out = sspaddmm(inp, mat1, mat2)
    ref = torch.sspaddmm(inp, mat1, mat2)
    assert out.dtype == torch.int64
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=0, atol=0)


def test_sspaddmm_empty_mat1_is_beta_times_input():
    torch.manual_seed(3)
    inp = torch.randn(3, 4).to_sparse().coalesce()
    empty = torch.sparse_coo_tensor(
        torch.zeros(2, 0, dtype=torch.int64), torch.zeros(0), (3, 5)
    )
    mat2 = torch.randn(5, 4, dtype=torch.float32)
    out = sspaddmm(inp, empty, mat2, beta=2.0, alpha=3.0)
    ref = torch.sspaddmm(inp, empty, mat2, beta=2.0, alpha=3.0)
    torch.testing.assert_close(out.to_dense(), 2.0 * inp.to_dense(), rtol=0, atol=0)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=0, atol=0)


def test_sspaddmm_uncoalesced_inputs_coalesced_first():
    dup_in = torch.sparse_coo_tensor(
        torch.tensor([[0, 0], [0, 1]]), torch.tensor([1.0, 2.0]), (2, 2)
    )
    dup_m1 = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]]), torch.tensor([3.0, 4.0]), (2, 2)
    )
    out = sspaddmm(dup_in, dup_m1, torch.eye(2))
    ref = torch.sspaddmm(dup_in, dup_m1, torch.eye(2))
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=0, atol=0)


def test_sspaddmm_non_contiguous_dense_mat():
    torch.manual_seed(4)
    base = torch.randn(4, 5, dtype=torch.float32)
    mat2 = base.t()  # (5, 4) non-contiguous view
    assert not mat2.is_contiguous()
    inp = torch.randn(3, 4).to_sparse().coalesce()
    mat1 = torch.randn(3, 5).to_sparse().coalesce()
    out = sspaddmm(inp, mat1, mat2)
    ref = torch.sspaddmm(inp, mat1, mat2)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-5, atol=1e-5)


def test_sspaddmm_errors_match_upstream_asserts():
    torch.manual_seed(5)
    dense = torch.randn(3, 4)
    sp = torch.randn(3, 4).to_sparse().coalesce()
    mat1 = torch.randn(3, 5).to_sparse().coalesce()
    mat2 = torch.randn(5, 4, dtype=torch.float32)
    with pytest.raises(AssertionError, match="input must be a sparse COO tensor"):
        sspaddmm(dense, mat1, mat2)
    with pytest.raises(AssertionError, match="mat1 must be a sparse COO tensor"):
        sspaddmm(sp, dense, mat2)
    with pytest.raises(AssertionError, match="mat2 must be a dense tensor"):
        sspaddmm(sp, mat1, mat1)


def test_sspaddmm_out_writes_buffer():
    torch.manual_seed(6)
    inp = torch.randn(3, 4).to_sparse().coalesce()
    mat1 = torch.randn(3, 5).to_sparse().coalesce()
    mat2 = torch.randn(5, 4, dtype=torch.float32)
    out = torch.empty(3, 4).to_sparse().coalesce()
    ret = sspaddmm_out(inp, mat1, mat2, beta=1.5, alpha=0.5, out=out)
    assert ret is out
    ref = torch.sspaddmm(inp, mat1, mat2, beta=1.5, alpha=0.5)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-5, atol=1e-5)


def test_sspaddmm_out_requires_out():
    torch.manual_seed(7)
    inp = torch.randn(3, 4).to_sparse().coalesce()
    mat1 = torch.randn(3, 5).to_sparse().coalesce()
    mat2 = torch.randn(5, 4, dtype=torch.float32)
    with pytest.raises(TypeError, match=r"sspaddmm\(\): out must be provided"):
        sspaddmm_out(inp, mat1, mat2)


def test_sspaddmm_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.sspaddmm")
    launches = []
    kernel = mod._sspaddmm_row_kernel  # kernel name per upstream
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(8)
    inp = torch.randn(3, 4).to_sparse().coalesce()
    mat1 = torch.randn(3, 5).to_sparse().coalesce()
    mat2 = torch.randn(5, 4, dtype=torch.float32)
    out = sspaddmm(inp, mat1, mat2)
    assert len(launches) >= 1
    torch.testing.assert_close(
        out.to_dense(),
        torch.sspaddmm(inp, mat1, mat2).to_dense(),
        rtol=1e-5,
        atol=1e-5,
    )
