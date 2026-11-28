import importlib

import pytest
import torch

from .hspmm import hspmm


def _coo_dense(dense, dtype):
    return dense.to(dtype).to_sparse().coalesce()


def test_hspmm_matches_aten_fp32():
    torch.manual_seed(0)
    mat1 = _coo_dense(torch.randn(3, 5), torch.float32)
    mat2 = torch.randn(5, 4, dtype=torch.float32)
    out = hspmm(mat1, mat2)
    ref = torch.hspmm(mat1, mat2)
    assert out.layout == torch.sparse_coo
    assert out.shape == (3, 4) and out.dtype == torch.float32
    # aten's CPU hspmm sums each row like a dense dgemm, while this port's
    # pinned per-entry structure sums in COO storage order on to_dense(); the
    # two round differently (measured 1.2e-07 here), so compare at fp32
    # rounding tolerance.
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-6, atol=1e-6)


def test_hspmm_matches_aten_second_shape():
    torch.manual_seed(1)
    mat1 = _coo_dense(torch.randn(7, 11), torch.float32)
    mat2 = torch.randn(11, 2, dtype=torch.float32)
    out = hspmm(mat1, mat2)
    ref = torch.hspmm(mat1, mat2)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-6, atol=1e-6)


def test_hspmm_per_entry_structure_follows_upstream():
    # Upstream keeps one output row per input nonzero: out values have shape
    # (nnz, N) and row k equals val_k * mat2[col_k], in input COO order.
    torch.manual_seed(2)
    mat1 = _coo_dense(torch.randn(4, 6), torch.float32)
    mat2 = torch.randn(6, 3, dtype=torch.float32)
    out = hspmm(mat1, mat2)
    rows_in, cols_in = mat1._indices()
    vals_in = mat1._values()
    nnz = mat1._nnz()
    assert out._nnz() == nnz
    assert out._values().shape == (nnz, 3)
    assert out._indices().shape == (1, nnz)
    assert out._indices()[0].tolist() == rows_in.tolist()
    expected = torch.stack([vals_in[k] * mat2[cols_in[k]] for k in range(nnz)])
    torch.testing.assert_close(out._values(), expected, rtol=0, atol=0)


def test_hspmm_row_with_multiple_entries_sums_on_to_dense():
    torch.manual_seed(3)
    mat1 = _coo_dense(torch.randn(2, 6), torch.float32)  # dense nonzeros: full rows
    mat2 = torch.randn(6, 4, dtype=torch.float32)
    out = hspmm(mat1, mat2)
    ref = torch.hspmm(mat1, mat2)
    # COO-order row sums vs dgemm-order row sums: fp32 rounding tolerance.
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        out.to_dense(), mat1.to_dense() @ mat2, rtol=1e-6, atol=1e-6
    )


def test_hspmm_fp64_and_int64_match_aten():
    torch.manual_seed(4)
    mat1d = _coo_dense(torch.randn(3, 5), torch.float64)
    mat2d = torch.randn(5, 4, dtype=torch.float64)
    outd = hspmm(mat1d, mat2d)
    assert outd.dtype == torch.float64
    # Same COO-order vs dgemm-order rounding gap as fp32, at fp64 scale.
    torch.testing.assert_close(
        outd.to_dense(), torch.hspmm(mat1d, mat2d).to_dense(), rtol=1e-12, atol=1e-12
    )
    mat1i = _coo_dense(torch.randn(3, 5).round(), torch.int64)
    mat2i = torch.randint(-3, 4, (5, 4), dtype=torch.int64)
    outi = hspmm(mat1i, mat2i)
    assert outi.dtype == torch.int64
    torch.testing.assert_close(
        outi.to_dense(), torch.hspmm(mat1i, mat2i).to_dense(), rtol=0, atol=0
    )


def test_hspmm_empty_sparse_input():
    torch.manual_seed(5)
    indices = torch.zeros(2, 0, dtype=torch.int64)
    values = torch.zeros(0, dtype=torch.float32)
    mat1 = torch.sparse_coo_tensor(indices, values, (3, 5))
    mat2 = torch.randn(5, 4, dtype=torch.float32)
    out = hspmm(mat1, mat2)
    ref = torch.hspmm(mat1, mat2)
    assert out._nnz() == 0
    assert out.shape == (3, 4)
    assert out._values().shape == (0, 4)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=0, atol=0)


def test_hspmm_non_contiguous_dense_mat():
    torch.manual_seed(6)
    base = torch.randn(4, 5, dtype=torch.float32)
    mat2 = base.t()  # (5, 4) non-contiguous view
    assert not mat2.is_contiguous()
    mat1 = _coo_dense(torch.randn(3, 5), torch.float32)
    out = hspmm(mat1, mat2)
    ref = torch.hspmm(mat1, mat2)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-6, atol=1e-6)


def test_hspmm_errors_match_upstream_validation():
    torch.manual_seed(7)
    dense = torch.randn(3, 5, dtype=torch.float32)
    csr = dense.to_sparse().coalesce().to_sparse_csr()
    mat2 = torch.randn(5, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="hspmm: mat1 must be sparse COO"):
        hspmm(csr, mat2)
    with pytest.raises(ValueError, match="hspmm: mat1 must be sparse COO"):
        hspmm(dense, mat2)
    coo = dense.to_sparse().coalesce()
    with pytest.raises(ValueError, match=r"hspmm: mat1.shape\[1\]=5 must match"):
        hspmm(coo, torch.randn(6, 4))


def test_hspmm_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.hspmm")
    launches = []
    kernel = mod.hspmm_kernel  # kernel name per upstream
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(8)
    mat1 = _coo_dense(torch.randn(3, 5), torch.float32)
    mat2 = torch.randn(5, 4, dtype=torch.float32)
    out = hspmm(mat1, mat2)
    assert len(launches) >= 1
    torch.testing.assert_close(
        out.to_dense(), torch.hspmm(mat1, mat2).to_dense(), rtol=1e-6, atol=1e-6
    )
