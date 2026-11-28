import importlib

import pytest
import torch

from ._sparse_mm_reduce_impl import _sparse_mm_reduce_impl


def _coo(rows, cols, vals, shape, dtype=torch.float32):
    idx = torch.tensor([rows, cols], dtype=torch.int64)
    v = torch.tensor(vals, dtype=dtype)
    return torch.sparse_coo_tensor(idx, v, shape).coalesce()


# oracle: aten._sparse_mm_reduce_impl runs on CPU for CSR inputs (probed on
# torch 2.15.0+cpu; the COO spelling raises NotImplementedError, so the direct
# oracle always feeds CSR, while the FlagGems op also accepts COO by
# converting — those cases are checked against the CSR oracle result).


@pytest.mark.parametrize("shape", [(3, 4, 5), (6, 8, 7)])
@pytest.mark.parametrize("reduce", ["sum", "mean", "amax", "amin"])
def test_sparse_mm_reduce_impl_csr_matches_aten(shape, reduce):
    torch.manual_seed(0)
    M, K, N = shape
    mask = torch.rand(M, K) < 0.4
    csr = (torch.randn(M, K) * mask).to_sparse().coalesce().to_sparse_csr()
    other = torch.randn(K, N)

    out, arg = _sparse_mm_reduce_impl(csr, other, reduce)
    ref, ref_arg = torch.ops.aten._sparse_mm_reduce_impl(csr, other, reduce)
    assert out.shape == (M, N)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    # aten's CPU backend returns an empty int64 arg tensor (upstream docstring
    # pins this exact contract).
    assert arg.dtype == torch.int64 and arg.numel() == 0
    assert ref_arg.numel() == 0


def test_sparse_mm_reduce_impl_coo_input_converted():
    torch.manual_seed(1)
    sp = _coo([0, 1, 2, 2, 2], [0, 1, 0, 1, 2], [1.0, 2.0, 3.0, -1.0, 4.0], (3, 3))
    other = torch.randn(3, 4)
    csr = sp.to_sparse_csr()
    for reduce in ("sum", "mean", "amax", "amin"):
        out, _ = _sparse_mm_reduce_impl(sp, other, reduce)
        ref, _ = torch.ops.aten._sparse_mm_reduce_impl(csr, other, reduce)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_sparse_mm_reduce_impl_mean_divides_by_stored_nnz():
    torch.manual_seed(2)
    # Row 0 has 3 stored entries, row 1 has 1: mean = sum / per-row nnz.
    sp = _coo([0, 0, 0, 1], [0, 1, 2, 1], [2.0, 4.0, 6.0, 5.0], (2, 3))
    other = torch.eye(3)
    out_sum, _ = _sparse_mm_reduce_impl(sp.to_sparse_csr(), other, "sum")
    out_mean, _ = _sparse_mm_reduce_impl(sp.to_sparse_csr(), other, "mean")
    torch.testing.assert_close(out_mean[0], out_sum[0] / 3.0, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out_mean[1], out_sum[1], rtol=1e-6, atol=1e-6)


def test_sparse_mm_reduce_impl_empty_rows_are_zero():
    torch.manual_seed(3)
    sp = _coo([0, 2], [0, 1], [1.0, -2.0], (3, 2))
    other = torch.randn(2, 3)
    for reduce in ("sum", "mean", "amax", "amin"):
        out, _ = _sparse_mm_reduce_impl(sp.to_sparse_csr(), other, reduce)
        assert torch.all(out[1] == 0), reduce


def test_sparse_mm_reduce_impl_single_entry():
    torch.manual_seed(4)
    sp = _coo([1], [2], [3.0], (4, 5))
    other = torch.randn(5, 6)
    out, _ = _sparse_mm_reduce_impl(sp.to_sparse_csr(), other, "amin")
    expected = torch.zeros(4, 6)
    expected[1] = 3.0 * other[2]
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_sparse_mm_reduce_impl_max_min_aliases():
    torch.manual_seed(5)
    csr = _coo([0, 1, 2, 2], [0, 1, 0, 2], [1.0, 2.0, 3.0, 4.0], (3, 3)).to_sparse_csr()
    other = torch.randn(3, 4)
    torch.testing.assert_close(
        _sparse_mm_reduce_impl(csr, other, "max")[0],
        _sparse_mm_reduce_impl(csr, other, "amax")[0],
    )
    torch.testing.assert_close(
        _sparse_mm_reduce_impl(csr, other, "min")[0],
        _sparse_mm_reduce_impl(csr, other, "amin")[0],
    )


def test_sparse_mm_reduce_impl_float64():
    torch.manual_seed(6)
    sp = _coo(
        [0, 1, 2, 2], [0, 1, 0, 2], [1.0, 2.0, 3.0, 4.0], (3, 3), dtype=torch.float64
    )
    other = torch.randn(3, 4).double()
    out, _ = _sparse_mm_reduce_impl(sp.to_sparse_csr(), other, "amax")
    ref, _ = torch.ops.aten._sparse_mm_reduce_impl(sp.to_sparse_csr(), other, "amax")
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-10)


def test_sparse_mm_reduce_impl_errors_match_upstream_messages():
    torch.manual_seed(7)
    csr = _coo([0, 1, 2, 2], [0, 1, 0, 2], [1.0, 2.0, 3.0, 4.0], (3, 3)).to_sparse_csr()
    other = torch.randn(3, 4)
    with pytest.raises(
        NotImplementedError, match="Could not run '_sparse_mm_reduce_impl'"
    ):
        _sparse_mm_reduce_impl(torch.randn(3, 3), other, "sum")
    with pytest.raises(RuntimeError, match="Expected other to have strided layout"):
        _sparse_mm_reduce_impl(csr, csr, "sum")
    with pytest.raises(RuntimeError, match="expected self to be a 2-D tensor"):
        _sparse_mm_reduce_impl(
            torch.randn(2, 3, 3).to_sparse().coalesce(), torch.randn(3, 4), "sum"
        )
    with pytest.raises(
        RuntimeError, match="reduce argument must be either sum, prod, mean"
    ):
        _sparse_mm_reduce_impl(csr, other, "std")
    with pytest.raises(RuntimeError, match="reduce type of prod has not been enabled"):
        _sparse_mm_reduce_impl(csr, other, "prod")
    with pytest.raises(RuntimeError, match="Expected a tensor of dimension 2"):
        _sparse_mm_reduce_impl(csr, torch.randn(4, 3).unsqueeze(0), "sum")
    with pytest.raises(RuntimeError, match="tensor.size\\[0\\] = 4"):
        _sparse_mm_reduce_impl(csr, torch.randn(4, 3), "sum")
    with pytest.raises(RuntimeError, match="'other' to have scalar type"):
        _sparse_mm_reduce_impl(csr, other.double(), "sum")
    with pytest.raises(RuntimeError, match='spmm_reduce_kernel" not implemented'):
        _sparse_mm_reduce_impl(
            csr.to(torch.int64), torch.zeros(3, 4, dtype=torch.int64), "sum"
        )


def test_sparse_mm_reduce_impl_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._sparse_mm_reduce_impl")
    launches = []
    kernel = mod._sparse_mm_reduce_impl_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    csr = _coo([0, 1, 2, 2], [0, 1, 0, 2], [1.0, 2.0, 3.0, 4.0], (3, 3)).to_sparse_csr()
    other = torch.randn(3, 4)
    out, _ = _sparse_mm_reduce_impl(csr, other, "sum")
    ref, _ = torch.ops.aten._sparse_mm_reduce_impl(csr, other, "sum")
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
