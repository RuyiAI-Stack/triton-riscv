import importlib

import pytest
import torch

from ._sparse_sparse_matmul import _sparse_sparse_matmul, _sparse_sparse_matmul_out

# fp16 products are compared against the fp32 reference (~1e-3 relative
# rounding); atol 1e-2 keeps margin for accumulation-order differences.


def _coo(rows, cols, vals, shape, dtype=torch.float32):
    idx = torch.tensor([rows, cols], dtype=torch.int64)
    v = torch.tensor(vals, dtype=dtype)
    return torch.sparse_coo_tensor(idx, v, shape).coalesce()


# oracle: torch._sparse_sparse_matmul runs on CPU (probed on torch
# 2.15.0+cpu): COO x COO -> coalesced row-major COO, structural zeros kept.


@pytest.mark.parametrize("shape", [(5, 5, 5), (8, 6, 7)])
def test_sparse_sparse_matmul_matches_torch(shape):
    torch.manual_seed(0)
    M, K, N = shape
    a = (torch.randn(M, K) * (torch.rand(M, K) < 0.4)).to_sparse().coalesce()
    b = (torch.randn(K, N) * (torch.rand(K, N) < 0.4)).to_sparse().coalesce()

    out = _sparse_sparse_matmul(a, b)
    ref = torch._sparse_sparse_matmul(a, b)
    assert out.shape == (M, N) and out.layout == torch.sparse_coo
    assert out.is_coalesced
    assert out._nnz() == ref._nnz()
    torch.testing.assert_close(
        out.coalesce().indices(), ref.coalesce().indices(), rtol=0, atol=0
    )
    # Entry values are fp32 sums over K: the contraction order differs between
    # codegens (CI measured a 1-ulp gap on riscv64), so exactness is not
    # portable; the dense comparison below keeps the tight 1e-5 band.
    torch.testing.assert_close(out.values(), ref.values(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-5, atol=1e-5)


def test_sparse_sparse_matmul_rectangular_and_dtype_matrix():
    torch.manual_seed(1)
    for dtype in (torch.float32, torch.float64, torch.float16):
        a = _coo([0, 1, 2, 2], [0, 1, 0, 2], [1.0, 2.0, 3.0, 4.0], (3, 3), dtype)
        b = _coo([0, 2, 1, 2], [1, 0, 2, 2], [-1.0, 0.5, 2.0, 1.5], (3, 4), dtype)
        out = _sparse_sparse_matmul(a, b)
        if dtype == torch.float16:
            # Low-precision accumulation rounding is part of the contract
            # (upstream docstring); aten's CPU oracle itself rejects Half
            # (NotImplementedError), so compare against the exact fp32
            # product of the same fp16 values instead.
            expected = a.to_dense().float() @ b.to_dense().float()
            torch.testing.assert_close(
                out.to_dense().float(), expected, rtol=1e-2, atol=1e-2
            )
        else:
            ref = torch._sparse_sparse_matmul(a, b)
            assert out._nnz() == ref._nnz()
            torch.testing.assert_close(
                out.to_dense(), ref.to_dense(), rtol=1e-5, atol=1e-5
            )


def test_sparse_sparse_matmul_structural_zero_preserved():
    # [[1, 1]] @ [[1], [-1]] keeps one stored entry with value 0 (the aten
    # contract quoted verbatim in the upstream docstring; probe-confirmed).
    a = _coo([0, 0], [0, 1], [1.0, 1.0], (1, 2))
    b = _coo([0, 1], [0, 0], [1.0, -1.0], (2, 1))
    out = _sparse_sparse_matmul(a, b)
    assert out._nnz() == 1
    assert out.values()[0].item() == 0.0
    assert out.is_coalesced


def test_sparse_sparse_matmul_output_row_major_sorted():
    torch.manual_seed(3)
    a = _coo([2, 0, 1], [1, 2, 0], [1.0, 2.0, 3.0], (3, 3))
    out = _sparse_sparse_matmul(a, a)
    ref = torch._sparse_sparse_matmul(a, a)
    idx = out.coalesce().indices()
    # Row-major sorted = (row, col) pairs in lexicographic order. Sorting the
    # index rows independently would break the row/col pairing (the oracle's
    # own column sequence [1, 2, 0] is per-row, not globally ascending).
    keys = idx[0] * out.shape[1] + idx[1]
    assert torch.equal(torch.argsort(keys), torch.arange(idx.shape[1]))
    torch.testing.assert_close(idx, ref.coalesce().indices(), rtol=0, atol=0)


def test_sparse_sparse_matmul_identity_operand():
    torch.manual_seed(4)
    eye = torch.eye(4).to_sparse().coalesce()
    a = (torch.randn(4, 4) * (torch.rand(4, 4) < 0.5)).to_sparse().coalesce()
    out = _sparse_sparse_matmul(a, eye)
    assert out._nnz() == a._nnz()
    torch.testing.assert_close(out.to_dense(), a.to_dense(), rtol=1e-5, atol=1e-5)


def test_sparse_sparse_matmul_empty_operands():
    e = torch.sparse_coo_tensor(
        torch.zeros(2, 0, dtype=torch.int64), torch.zeros(0), (3, 3)
    ).coalesce()
    a = _coo([0, 1], [0, 1], [1.0, 2.0], (3, 3))
    out = _sparse_sparse_matmul(e, e)
    assert out._nnz() == 0 and out.shape == (3, 3)
    out2 = _sparse_sparse_matmul(a, e)
    assert out2._nnz() == 0


def test_sparse_sparse_matmul_out_writes_result_into_buffer():
    torch.manual_seed(5)
    a = _coo([0, 1, 2, 2], [0, 1, 0, 2], [1.0, 2.0, 3.0, 4.0], (3, 3))
    b = _coo([0, 1, 2], [2, 0, 1], [1.0, -1.0, 2.0], (3, 3))
    ref = _sparse_sparse_matmul(a, b)
    out = torch.empty_like(ref)
    ret = _sparse_sparse_matmul_out(a, b, out=out)
    assert ret is out
    torch.testing.assert_close(
        out.coalesce().indices(), ref.coalesce().indices(), rtol=0, atol=0
    )
    # Entry values are fp32 sums over K: the contraction order differs between
    # codegens (CI measured a 1-ulp gap on riscv64), so exactness is not
    # portable; the dense comparison below keeps the tight 1e-5 band.
    torch.testing.assert_close(out.values(), ref.values(), rtol=1e-6, atol=1e-6)


def test_sparse_sparse_matmul_out_rejects_mismatched_buffers():
    torch.manual_seed(6)
    a = _coo([0, 1], [0, 1], [1.0, 2.0], (2, 2))
    b = _coo([0, 1], [1, 0], [3.0, 4.0], (2, 2))
    ref = _sparse_sparse_matmul(a, b)
    with pytest.raises(
        RuntimeError, match="Expected out tensor to have sparse COO layout"
    ):
        _sparse_sparse_matmul_out(a, b, out=ref.to_sparse_csr())
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        _sparse_sparse_matmul_out(a, b, out=torch.empty_like(ref, dtype=torch.float64))
    with pytest.raises(RuntimeError, match="Expected out tensor to have shape"):
        _sparse_sparse_matmul_out(a, b, out=torch.empty((3, 3)).to_sparse())


def test_sparse_sparse_matmul_errors_match_upstream_messages():
    torch.manual_seed(7)
    a = _coo([0, 1], [0, 1], [1.0, 2.0], (2, 3))
    b = _coo([0, 1], [1, 0], [3.0, 4.0], (3, 2))
    with pytest.raises(RuntimeError, match="expected sparse tensors for both operands"):
        _sparse_sparse_matmul(a.to_dense(), b)
    with pytest.raises(RuntimeError, match="expected 2-D sparse tensors"):
        _sparse_sparse_matmul(
            torch.randn(2, 2, 2).to_sparse().coalesce(),
            torch.randn(2, 2, 2).to_sparse(),
        )
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        _sparse_sparse_matmul(a, a)
    with pytest.raises(RuntimeError, match="does not match mat2 dtype"):
        _sparse_sparse_matmul(a, b.double())
    with pytest.raises(NotImplementedError, match='sparse_matmul" not implemented'):
        _sparse_sparse_matmul(a.to(torch.int64), b.to(torch.int64))


def test_sparse_sparse_matmul_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._sparse_sparse_matmul")
    launches = []
    kernel = mod._sspm_accum
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    a = (torch.randn(5, 5) * (torch.rand(5, 5) < 0.4)).to_sparse().coalesce()
    b = (torch.randn(5, 5) * (torch.rand(5, 5) < 0.4)).to_sparse().coalesce()
    out = _sparse_sparse_matmul(a, b)
    ref = torch._sparse_sparse_matmul(a, b)
    assert len(launches) >= 1
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=1e-5, atol=1e-5)
