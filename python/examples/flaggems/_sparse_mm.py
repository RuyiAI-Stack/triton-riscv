import torch
import triton
import triton.language as tl


_SPARSE_MM_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}

# Reduction selectors accepted by ``aten::_sparse_mm.reduce``.  torch maps the
# ``max``/``min`` spellings onto amax/amin in the same kernel; we normalise them
# to a single code so the Triton kernel only needs one constexpr switch.
_REDUCE_CODES = {
    "sum": 0,
    "mean": 1,
    "amax": 2,
    "amin": 3,
    "max": 2,
    "min": 3,
}

_REDUCE_SUM = tl.constexpr(0)
_REDUCE_MEAN = tl.constexpr(1)
_REDUCE_AMAX = tl.constexpr(2)
_REDUCE_AMIN = tl.constexpr(3)


@triton.jit
def _sparse_mm_kernel(
    crow_ptr,
    col_ptr,
    val_ptr,
    mat2_ptr,
    out_ptr,
    N,
    stride_mat2_k,
    stride_mat2_n,
    stride_out_m,
    stride_out_n,
    IS_FP64: tl.constexpr,
    IS_REDUCE: tl.constexpr,
    REDUCE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per (row, column block) of the dense output.

    Walks the CSR row ``[crow[row], crow[row + 1])`` and accumulates
    ``val[e] * mat2[col[e], :]`` in fp32/fp64.  For the plain ``_sparse_mm``
    path the accumulated sum is written directly; for the ``.reduce`` overload
    the per-nnz products are folded with the requested reduction (rows with no
    stored entries yield zeros for every reduction, matching torch).
    """
    row = tl.program_id(0)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rn < N
    row64 = row.to(tl.int64)
    col_off = rn.to(tl.int64) * stride_mat2_n

    acc_dtype = tl.float64 if IS_FP64 else tl.float32
    acc = tl.zeros((BLOCK_N,), dtype=acc_dtype)

    start = tl.load(crow_ptr + row64).to(tl.int64)
    end = tl.load(crow_ptr + row64 + 1).to(tl.int64)

    if IS_REDUCE:
        if REDUCE == _REDUCE_AMAX or REDUCE == _REDUCE_AMIN:
            # Fold with tl.where so the first stored product seeds acc without a
            # dynamic branch; max/min over an empty row collapses to zero below.
            for e in range(start, end):
                c = tl.load(col_ptr + e).to(tl.int64)
                v = tl.load(val_ptr + e).to(acc_dtype)
                b = tl.load(
                    mat2_ptr + c * stride_mat2_k + col_off, mask=rmask, other=0.0
                ).to(acc_dtype)
                p = b * v
                if REDUCE == _REDUCE_AMAX:
                    acc = tl.where(e == start, p, tl.maximum(acc, p))
                else:
                    acc = tl.where(e == start, p, tl.minimum(acc, p))
            acc = tl.where(start < end, acc, 0.0)
        else:
            for e in range(start, end):
                c = tl.load(col_ptr + e).to(tl.int64)
                v = tl.load(val_ptr + e).to(acc_dtype)
                b = tl.load(
                    mat2_ptr + c * stride_mat2_k + col_off, mask=rmask, other=0.0
                ).to(acc_dtype)
                acc += b * v
            if REDUCE == _REDUCE_MEAN:
                nnz = (end - start).to(acc_dtype)
                acc = tl.where(nnz > 0, acc / nnz, 0.0)
    else:
        for e in range(start, end):
            c = tl.load(col_ptr + e).to(tl.int64)
            v = tl.load(val_ptr + e).to(acc_dtype)
            b = tl.load(
                mat2_ptr + c * stride_mat2_k + col_off, mask=rmask, other=0.0
            ).to(acc_dtype)
            acc += b * v

    # Widen before the multiply: ``rn * stride_out_n`` in int32 can wrap for
    # large N * stride_out_n (the flat offset itself already needs int64).
    cos = row64 * stride_out_m + rn.to(tl.int64) * stride_out_n
    tl.store(out_ptr + cos, acc.to(out_ptr.dtype.element_ty), mask=rmask)


def _to_csr(sparse):
    if sparse.layout == torch.sparse_csr:
        return sparse
    if sparse.layout in (torch.sparse_coo, torch.sparse_csc):
        return sparse.to_sparse_csr()
    if sparse.layout in (torch.sparse_bsr, torch.sparse_bsc):
        # ``to_sparse_csr`` only accepts CSR/CSC sources; BSR/BSC go through
        # COO, which both layouts can produce.
        return sparse.to_sparse().to_sparse_csr()
    raise RuntimeError(
        f"_sparse_mm: Expected sparse to have a sparse layout, but got {sparse.layout}"
    )


def _sparse_mm_impl(sparse, dense, reduce=None):
    if sparse.layout == torch.strided:
        raise RuntimeError(
            f"_sparse_mm: Expected sparse to have a sparse layout, but got "
            f"{sparse.layout}"
        )
    if dense.layout != torch.strided:
        raise RuntimeError(
            f"_sparse_mm: Expected dense to have strided layout, but got {dense.layout}"
        )

    if sparse.dim() != 2:
        raise RuntimeError(
            f"_sparse_mm: Expected sparse to be a 2-D matrix, but got "
            f"{sparse.dim()}-D tensor"
        )
    if dense.dim() != 2:
        raise RuntimeError(
            f"_sparse_mm: Expected dense to be a 2-D matrix, but got "
            f"{dense.dim()}-D tensor"
        )
    if sparse.dense_dim() != 0:
        raise RuntimeError("_sparse_mm: Expected non-hybrid sparse tensor")

    if sparse.dtype not in _SPARSE_MM_DTYPES:
        raise RuntimeError(
            f"_sparse_mm: Expected sparse to be floating-point, but got {sparse.dtype}"
        )
    if sparse.dtype != dense.dtype:
        raise RuntimeError(
            f"_sparse_mm: Expected both inputs to have the same dtype, but got "
            f"sparse={sparse.dtype}, dense={dense.dtype}"
        )

    M, K = sparse.shape
    if dense.shape[0] != K:
        raise RuntimeError("_sparse_mm: sparse and dense shapes cannot be multiplied")
    N = dense.shape[1]

    if reduce is not None:
        if reduce not in _REDUCE_CODES:
            raise RuntimeError(
                f"_sparse_mm.reduce: reduce argument must be either sum, prod, "
                f"mean, amax or amin, got {reduce}"
            )
        if sparse.layout != torch.sparse_csr:
            raise RuntimeError(
                f"_sparse_mm.reduce: expect sparse to be SparseCsr, got {sparse.layout}"
            )

    out = torch.empty((M, N), dtype=dense.dtype, device=dense.device)
    if M == 0 or N == 0:
        return out

    csr = _to_csr(sparse)
    crow = csr.crow_indices()
    col = csr.col_indices()
    val = csr.values()

    block_n = min(triton.next_power_of_2(N), 256)
    grid = (M, triton.cdiv(N, block_n))
    _sparse_mm_kernel[grid](
        crow,
        col,
        val,
        dense,
        out,
        N,
        dense.stride(0),
        dense.stride(1),
        out.stride(0),
        out.stride(1),
        IS_FP64=dense.dtype == torch.float64,
        IS_REDUCE=reduce is not None,
        REDUCE=_REDUCE_CODES[reduce] if reduce is not None else 0,
        BLOCK_N=block_n,
    )
    return out


def _sparse_mm(sparse, dense, reduce=None):
    """SpMM with an optional reduction over the per-nnz products.

    Registered for both ``aten::_sparse_mm`` and ``aten::_sparse_mm.reduce``;
    the latter passes the reduction selector as a third positional argument
    (the same one-wrapper-for-many-overloads pattern as ``scatter``).
    """
    return _sparse_mm_impl(sparse, dense, reduce=reduce)
