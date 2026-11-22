import importlib

import pytest
import torch

from ._sparse_mm import _sparse_mm


def _sparse_coo(rows, cols, vals, shape):
    idx = torch.tensor([rows, cols], dtype=torch.int64)
    return torch.sparse_coo_tensor(
        idx, torch.tensor(vals, dtype=torch.float32), shape
    ).coalesce()


# oracle: aten._sparse_mm runs on CPU for plain COO/CSR SpMM (probed on torch
# 2.15.0+cpu); the .reduce overload delegates to aten._sparse_mm_reduce_impl,
# which runs on CPU for CSR inputs — the same composition the upstream
# FlagGems wrapper itself uses for its CPU fallback.


@pytest.mark.parametrize("shape", [(3, 4, 5), (8, 16, 12)])
def test_sparse_mm_coo_matches_aten(shape):
    torch.manual_seed(0)
    M, K, N = shape
    density = 0.4
    mask = torch.rand(M, K) < density
    dense = torch.randn(M, K) * mask
    sp = dense.to_sparse().coalesce()
    mat2 = torch.randn(K, N)

    out = _sparse_mm(sp, mat2)
    ref = torch.ops.aten._sparse_mm(sp, mat2)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_sparse_mm_csr_matches_aten():
    torch.manual_seed(1)
    sp = _sparse_coo(
        [0, 1, 2, 2], [0, 1, 0, 2], [1.0, 2.0, 3.0, 4.0], (3, 3)
    ).to_sparse_csr()
    mat2 = torch.randn(3, 4)
    out = _sparse_mm(sp, mat2)
    ref = torch.ops.aten._sparse_mm(sp, mat2)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_sparse_mm_csc_layout_accepted():
    torch.manual_seed(2)
    sp = _sparse_coo(
        [0, 1, 2, 2], [0, 1, 0, 2], [1.0, 2.0, 3.0, 4.0], (3, 3)
    ).to_sparse_csc()
    mat2 = torch.randn(3, 4)
    out = _sparse_mm(sp, mat2)
    ref = sp.to_dense() @ mat2
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_sparse_mm_float64_and_float16():
    torch.manual_seed(3)
    for dtype in (torch.float64, torch.float16):
        sp = (
            _sparse_coo([0, 1, 2, 2], [0, 1, 0, 2], [1.0, 2.0, 3.0, 4.0], (3, 3))
            .to(dtype)
            .coalesce()
        )
        mat2 = torch.randn(3, 4).to(dtype)
        out = _sparse_mm(sp, mat2)
        ref = torch.ops.aten._sparse_mm(sp, mat2)
        torch.testing.assert_close(out.double(), ref.double(), rtol=1e-3, atol=1e-3)


def test_sparse_mm_empty_rows_yield_zeros():
    torch.manual_seed(4)
    sp = _sparse_coo([0, 2], [0, 1], [1.0, -2.0], (3, 2))
    mat2 = torch.randn(2, 3)
    out = _sparse_mm(sp, mat2)
    assert torch.all(out[1] == 0)
    torch.testing.assert_close(out, sp.to_dense() @ mat2, rtol=1e-5, atol=1e-5)


def test_sparse_mm_non_contiguous_dense():
    torch.manual_seed(5)
    sp = _sparse_coo([0, 1, 2, 2], [0, 1, 0, 2], [1.0, 2.0, 3.0, 4.0], (3, 3))
    mat2 = torch.randn(4, 3).t()  # (3, 4) view with strides (1, 4)
    assert not mat2.is_contiguous()
    out = _sparse_mm(sp, mat2)
    torch.testing.assert_close(out, sp.to_dense() @ mat2, rtol=1e-5, atol=1e-5)


def test_sparse_mm_empty_output_dim():
    torch.manual_seed(6)
    sp = _sparse_coo([0], [0], [1.0], (3, 2))
    mat2 = torch.randn(2, 0)
    out = _sparse_mm(sp, mat2)
    assert out.shape == (3, 0)


@pytest.mark.parametrize("reduce", ["sum", "mean", "amax", "amin"])
def test_sparse_mm_reduce_matches_reduce_impl(reduce):
    # aten's reduce overload has no COO CPU kernel; compare against the CSR
    # _sparse_mm_reduce_impl backend (upstream _sparse_mm delegates to it too).
    torch.manual_seed(7)
    sp = _sparse_coo(
        [0, 1, 2, 2, 2], [0, 1, 0, 1, 2], [1.0, 2.0, 3.0, -1.0, 4.0], (3, 3)
    )
    mat2 = torch.randn(3, 4)
    csr = sp.to_sparse_csr()

    # reduce requires CSR (pinned by the error test below), so the impl call
    # consumes the CSR form; the oracle is the same CSR composition.
    out = _sparse_mm(csr, mat2, reduce=reduce)
    ref, _ = torch.ops.aten._sparse_mm_reduce_impl(csr, mat2, reduce)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_sparse_mm_reduce_max_min_aliases():
    torch.manual_seed(8)
    sp = _sparse_coo([0, 1, 2, 2], [0, 1, 0, 2], [1.0, 2.0, 3.0, 4.0], (3, 3))
    mat2 = torch.randn(3, 4)
    csr = sp.to_sparse_csr()
    torch.testing.assert_close(
        _sparse_mm(csr, mat2, reduce="max"), _sparse_mm(csr, mat2, reduce="amax")
    )
    torch.testing.assert_close(
        _sparse_mm(csr, mat2, reduce="min"), _sparse_mm(csr, mat2, reduce="amin")
    )


def test_sparse_mm_reduce_empty_rows_are_zero():
    torch.manual_seed(9)
    sp = _sparse_coo([0, 2], [0, 1], [1.0, -2.0], (3, 2)).to_sparse_csr()
    mat2 = torch.randn(2, 3)
    for reduce in ("sum", "mean", "amax", "amin"):
        out = _sparse_mm(sp, mat2, reduce=reduce)
        assert torch.all(out[1] == 0), reduce


def test_sparse_mm_errors_match_upstream_messages():
    torch.manual_seed(10)
    sp = _sparse_coo([0, 1, 2, 2], [0, 1, 0, 2], [1.0, 2.0, 3.0, 4.0], (3, 3))
    mat2 = torch.randn(3, 4)
    with pytest.raises(RuntimeError, match="Expected sparse to have a sparse layout"):
        _sparse_mm(mat2, mat2)
    with pytest.raises(RuntimeError, match="Expected dense to have strided layout"):
        _sparse_mm(sp, sp)
    with pytest.raises(RuntimeError, match="Expected sparse to be a 2-D matrix"):
        _sparse_mm(sp.to_dense().unsqueeze(0).to_sparse(), mat2)
    with pytest.raises(
        RuntimeError, match="Expected both inputs to have the same dtype"
    ):
        _sparse_mm(sp, mat2.double())
    with pytest.raises(RuntimeError, match="Expected sparse to be floating-point"):
        _sparse_mm(sp.to(torch.int64), torch.zeros(3, 4, dtype=torch.int64))
    with pytest.raises(
        RuntimeError, match="sparse and dense shapes cannot be multiplied"
    ):
        _sparse_mm(sp, torch.randn(4, 4))
    with pytest.raises(
        RuntimeError,
        match="reduce argument must be either sum, prod, mean, amax or amin",
    ):
        _sparse_mm(sp, mat2, reduce="prod")
    with pytest.raises(RuntimeError, match="expect sparse to be SparseCsr"):
        _sparse_mm(sp, mat2, reduce="sum")


def test_sparse_mm_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._sparse_mm")
    launches = []
    kernel = mod._sparse_mm_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    sp = _sparse_coo([0, 1, 2, 2], [0, 1, 0, 2], [1.0, 2.0, 3.0, 4.0], (3, 3))
    mat2 = torch.randn(3, 4)
    out = _sparse_mm(sp, mat2)
    ref = torch.ops.aten._sparse_mm(sp, mat2)
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
