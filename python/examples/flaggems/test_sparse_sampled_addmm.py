import importlib

import pytest
import torch

from .sparse_sampled_addmm import sparse_sampled_addmm, sparse_sampled_addmm_out


def _csr(dense, dtype=torch.float32):
    return dense.to(dtype).to_sparse().coalesce().to_sparse_csr()


def test_sampled_addmm_matches_torch_two_shapes():
    torch.manual_seed(0)
    for m, k, n in ((3, 5, 4), (6, 8, 2)):
        inp = _csr(torch.randn(m, n))
        mat1 = torch.randn(m, k, dtype=torch.float32)
        mat2 = torch.randn(k, n, dtype=torch.float32)
        out = sparse_sampled_addmm(inp, mat1, mat2, beta=0.5, alpha=2.0)
        ref = torch.sparse.sampled_addmm(inp, mat1, mat2, beta=0.5, alpha=2.0)
        assert out.layout == torch.sparse_csr
        assert out.shape == (m, n) and out.dtype == torch.float32
        assert out._nnz() == ref._nnz() == inp._nnz()
        torch.testing.assert_close(
            out.crow_indices(), ref.crow_indices(), rtol=0, atol=0
        )
        torch.testing.assert_close(out.col_indices(), ref.col_indices(), rtol=0, atol=0)
        # 1e-5: fp32 dot order differs from torch's kernel.
        torch.testing.assert_close(out.values(), ref.values(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-5, atol=1e-5)


def test_sampled_addmm_masked_positions_stay_structural_zero():
    torch.manual_seed(1)
    base = torch.tensor([[1.0, 0, 2, 0], [0, 0, 0, 3], [4, 5, 0, 0]])
    inp = _csr(base)
    mat1 = torch.randn(3, 6, dtype=torch.float32)
    mat2 = torch.randn(6, 4, dtype=torch.float32)
    out = sparse_sampled_addmm(inp, mat1, mat2)
    dense = out.to_dense()
    mask = base != 0
    # Off-mask positions carry no stored values: to_dense is exactly zero there.
    assert torch.count_nonzero(dense[~mask]) == 0
    ref = torch.sparse.sampled_addmm(inp, mat1, mat2)
    torch.testing.assert_close(dense, ref.to_dense(), rtol=1e-5, atol=1e-5)


def test_sampled_addmm_fp64_arange_exact():
    inp = _csr(torch.tensor([[1.0, 0, 2], [0, 3, 0]]), dtype=torch.float64)
    mat1 = torch.arange(2 * 4, dtype=torch.float64).reshape(2, 4) / 4
    mat2 = torch.arange(4 * 3, dtype=torch.float64).reshape(4, 3) / 3
    out = sparse_sampled_addmm(inp, mat1, mat2, beta=2.0, alpha=3.0)
    ref = torch.sparse.sampled_addmm(inp, mat1, mat2, beta=2.0, alpha=3.0)
    torch.testing.assert_close(out.values(), ref.values(), rtol=0, atol=0)


def test_sampled_addmm_alpha_zero_returns_beta_times_input():
    torch.manual_seed(2)
    inp = _csr(torch.randn(3, 4))
    mat1 = torch.randn(3, 5, dtype=torch.float32)
    mat2 = torch.randn(5, 4, dtype=torch.float32)
    out = sparse_sampled_addmm(inp, mat1, mat2, beta=2.0, alpha=0.0)
    torch.testing.assert_close(out.values(), 2.0 * inp.values(), rtol=0, atol=0)
    ref = torch.sparse.sampled_addmm(inp, mat1, mat2, beta=2.0, alpha=0.0)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=0, atol=0)


def test_sampled_addmm_default_beta_alpha_match_torch():
    torch.manual_seed(3)
    inp = _csr(torch.randn(4, 6))
    mat1 = torch.randn(4, 5, dtype=torch.float32)
    mat2 = torch.randn(5, 6, dtype=torch.float32)
    out = sparse_sampled_addmm(inp, mat1, mat2)
    ref = torch.sparse.sampled_addmm(inp, mat1, mat2)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-5, atol=1e-5)


def test_sampled_addmm_empty_mask():
    torch.manual_seed(4)
    inp = _csr(torch.zeros(3, 4))  # nnz == 0
    assert inp._nnz() == 0
    mat1 = torch.randn(3, 5, dtype=torch.float32)
    mat2 = torch.randn(5, 4, dtype=torch.float32)
    out = sparse_sampled_addmm(inp, mat1, mat2)
    ref = torch.sparse.sampled_addmm(inp, mat1, mat2)
    assert out._nnz() == 0
    assert out.shape == (3, 4)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=0, atol=0)


def test_sampled_addmm_non_contiguous_dense_operands():
    torch.manual_seed(5)
    inp = _csr(torch.randn(3, 4))
    b1 = torch.randn(5, 3, dtype=torch.float32)
    mat1 = b1.t()  # (3, 5) non-contiguous view
    assert not mat1.is_contiguous()
    b2 = torch.randn(4, 5, dtype=torch.float32)
    mat2 = b2.t()  # (5, 4) non-contiguous view
    assert not mat2.is_contiguous()
    out = sparse_sampled_addmm(inp, mat1, mat2)
    ref = torch.sparse.sampled_addmm(inp, mat1, mat2)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-5, atol=1e-5)


def test_sampled_addmm_errors_match_upstream_validation():
    torch.manual_seed(6)
    inp = _csr(torch.randn(3, 4))
    mat1 = torch.randn(3, 5, dtype=torch.float32)
    mat2 = torch.randn(5, 4, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Expected input to have sparse csr layout"):
        sparse_sampled_addmm(torch.randn(3, 4).to_sparse().coalesce(), mat1, mat2)
    with pytest.raises(RuntimeError, match="Expected mat1 to have strided layout"):
        sparse_sampled_addmm(inp, torch.randn(3, 5).to_sparse().coalesce(), mat2)
    with pytest.raises(RuntimeError, match="Expected mat2 to have strided layout"):
        sparse_sampled_addmm(inp, mat1, torch.randn(5, 4).to_sparse().coalesce())
    with pytest.raises(
        RuntimeError, match="Expected all inputs to have the same dtype"
    ):
        sparse_sampled_addmm(inp, mat1.to(torch.float64), mat2.to(torch.float64))
    with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
        sparse_sampled_addmm(inp, mat1, torch.randn(6, 4))
    with pytest.raises(RuntimeError, match=r"input.shape\[-2:\] must match"):
        sparse_sampled_addmm(inp, mat1, torch.randn(5, 6))


def test_sampled_addmm_out_writes_matching_structure():
    torch.manual_seed(7)
    inp = _csr(torch.randn(3, 4))
    mat1 = torch.randn(3, 5, dtype=torch.float32)
    mat2 = torch.randn(5, 4, dtype=torch.float32)
    out = _csr(torch.randn(3, 4))
    ret = sparse_sampled_addmm_out(inp, mat1, mat2, beta=1.5, alpha=2.5, out=out)
    assert ret is out
    ref = torch.sparse.sampled_addmm(inp, mat1, mat2, beta=1.5, alpha=2.5)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-5, atol=1e-5)


def test_sampled_addmm_out_requires_out():
    torch.manual_seed(8)
    inp = _csr(torch.randn(3, 4))
    mat1 = torch.randn(3, 5, dtype=torch.float32)
    mat2 = torch.randn(5, 4, dtype=torch.float32)
    with pytest.raises(
        TypeError, match=r"sparse_sampled_addmm\(\): out must be provided"
    ):
        sparse_sampled_addmm_out(inp, mat1, mat2)


def test_sampled_addmm_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.sparse_sampled_addmm")
    launches = []
    kernel = mod.sparse_sampled_addmm_kernel  # kernel name per upstream
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(9)
    inp = _csr(torch.randn(4, 6))
    mat1 = torch.randn(4, 5, dtype=torch.float32)
    mat2 = torch.randn(5, 6, dtype=torch.float32)
    out = sparse_sampled_addmm(inp, mat1, mat2)
    assert len(launches) >= 1
    ref = torch.sparse.sampled_addmm(inp, mat1, mat2)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-5, atol=1e-5)
