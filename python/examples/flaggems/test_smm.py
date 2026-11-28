import importlib

import pytest
import torch

from .smm import smm


def _sparse(dense, dtype=torch.float32):
    return dense.to(dtype).to_sparse().coalesce()


def test_smm_matches_torch_two_shapes():
    torch.manual_seed(0)
    for sm, sk, n in ((3, 5, 4), (7, 11, 2)):
        sp = _sparse(torch.randn(sm, sk))
        dense = torch.randn(sk, n, dtype=torch.float32)
        out = smm(sp, dense)
        ref = torch.smm(sp, dense).coalesce()
        assert out.layout == torch.sparse_coo
        assert out.shape == (sm, n) and out.dtype == torch.float32
        out_c = out.coalesce() if not out.is_coalesced() else out
        torch.testing.assert_close(
            out_c.indices(), ref.indices(), rtol=0, atol=0, msg=f"{(sm, sk, n)}"
        )
        # 1e-5: per-row fp32 summation order differs from torch's kernel.
        torch.testing.assert_close(out_c.values(), ref.values(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-5, atol=1e-5)


def test_smm_full_rows_materialize_all_columns():
    # A fully nonzero sparse input makes every structural row contribute all
    # N output columns (M * N output entries, ascending row/col order).
    torch.manual_seed(1)
    sp = _sparse(torch.randn(3, 5))
    dense = torch.randn(5, 4, dtype=torch.float32)
    out = smm(sp, dense)
    out_c = out.coalesce() if not out.is_coalesced() else out
    assert out_c._nnz() == 3 * 4
    rows, cols = out_c.indices()
    assert rows.tolist() == sorted(rows.tolist())
    assert set(rows.tolist()) == {0, 1, 2}
    torch.testing.assert_close(
        out.to_dense(), torch.smm(sp, dense).to_dense(), rtol=1e-5, atol=1e-5
    )


def test_smm_row_without_nonzeros_is_absent():
    # torch.smm only materializes rows present in the sparse structure.
    torch.manual_seed(2)
    dense_mask = torch.tensor([[1.0, 0, 0, 0, 2], [0, 0, 0, 0, 0], [0, 0, 3, 0, 0]])
    sp = _sparse(dense_mask)
    dense = torch.randn(5, 4, dtype=torch.float32)
    out = smm(sp, dense)
    ref = torch.smm(sp, dense).coalesce()
    out_c = out.coalesce() if not out.is_coalesced() else out
    assert out_c._nnz() == ref._nnz() == 2 * 4
    assert set(out_c.indices()[0].tolist()) == {0, 2}
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-5, atol=1e-5)


def test_smm_uncoalesced_input_duplicates_summed():
    dup = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1], [0, 1, 0]]), torch.tensor([1.0, 2.0, 3.0]), (2, 2)
    )
    assert not dup.is_coalesced()
    out = smm(dup, torch.eye(2))
    ref = torch.smm(dup, torch.eye(2)).coalesce()
    assert out.to_dense().shape == (2, 2)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=0, atol=0)


def test_smm_empty_sparse_input():
    torch.manual_seed(3)
    indices = torch.zeros(2, 0, dtype=torch.int64)
    values = torch.zeros(0, dtype=torch.float32)
    sp = torch.sparse_coo_tensor(indices, values, (3, 5))
    dense = torch.randn(5, 4, dtype=torch.float32)
    out = smm(sp, dense)
    ref = torch.smm(sp, dense)
    assert out._nnz() == 0
    assert out.shape == (3, 4)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=0, atol=0)


def test_smm_non_contiguous_dense_mat():
    torch.manual_seed(4)
    base = torch.randn(4, 5, dtype=torch.float32)
    dense = base.t()  # (5, 4) non-contiguous view
    assert not dense.is_contiguous()
    sp = _sparse(torch.randn(3, 5))
    out = smm(sp, dense)
    torch.testing.assert_close(
        out.to_dense(), torch.smm(sp, dense).to_dense(), rtol=1e-5, atol=1e-5
    )


def test_smm_errors_match_upstream_validation():
    torch.manual_seed(5)
    dense = torch.randn(3, 5, dtype=torch.float32)
    sp = _sparse(dense)
    mat = torch.randn(5, 4, dtype=torch.float32)
    with pytest.raises(RuntimeError, match=r"can only be called on sparse tensors"):
        smm(dense, mat)
    with pytest.raises(RuntimeError, match=r"matrices expected, got 3D tensor"):
        smm(_sparse(torch.randn(2, 2, 2)), mat)
    with pytest.raises(
        RuntimeError, match=r"sspaddmm: Argument #3: Expected dim 0 size 5, got 6"
    ):
        smm(sp, torch.randn(6, 4))
    with pytest.raises(RuntimeError, match="GEMS smm is only implemented for float32"):
        smm(sp.to(torch.float64), torch.randn(5, 4, dtype=torch.float64))
    with pytest.raises(RuntimeError, match="GEMS smm is only implemented for float32"):
        # torch.randn has no int64 overload (NotImplementedError), so the
        # dense operand is materialized as float first and cast.
        smm(sp.to(torch.int64), torch.randn(5, 4).to(torch.int64))


def test_smm_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.smm")
    launches = []
    kernel = mod.smm_coo_kernel  # kernel name per upstream
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(6)
    sp = _sparse(torch.randn(3, 5))
    dense = torch.randn(5, 4, dtype=torch.float32)
    out = smm(sp, dense)
    assert len(launches) >= 1
    torch.testing.assert_close(
        out.to_dense(), torch.smm(sp, dense).to_dense(), rtol=1e-5, atol=1e-5
    )
