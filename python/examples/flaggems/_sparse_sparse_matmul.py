import torch
import triton
import triton.language as tl


_SPARSE_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


@triton.jit
def _sspm_hist(idx_ptr, cnt_ptr, nnz, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < nnz
    r = tl.load(idx_ptr + offs, mask=m, other=0)
    tl.atomic_add(cnt_ptr + r, 1, mask=m)


@triton.jit
def _sspm_scan(cnt_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # Exclusive prefix sum: out[0] = 0, out[i + 1] = out[i] + cnt[i].
    # Single program, chunked over n.
    tl.store(out_ptr + tl.arange(0, 1), tl.zeros([1], dtype=tl.int32))
    carry = tl.zeros([1], dtype=tl.int32)
    for s in range(0, n, BLOCK):
        offs = s + tl.arange(0, BLOCK)
        m = offs < n
        c = tl.load(cnt_ptr + offs, mask=m, other=0)
        cs = tl.cumsum(c, axis=0)
        total = carry + cs
        tl.store(out_ptr + 1 + offs, total, mask=m)
        carry = carry + tl.sum(c, axis=0)


@triton.jit
def _sspm_scatter(
    row_ptr,
    col_ptr,
    val_ptr,
    cur_ptr,
    bptr,
    ocol_ptr,
    oval_ptr,
    nnz,
    BLOCK: tl.constexpr,
):
    # cur_ptr holds the per-row histogram counts; decrementing it yields
    # cnt-1, cnt-2, ... via the returned old values cnt, cnt-1, ..., so
    # slot = bptr[r + 1] - old covers the row segment exactly (old = cnt maps
    # to the segment start, old = 1 maps to the last slot) and leaves
    # cur_ptr back at zero for the next call.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < nnz
    r = tl.load(row_ptr + offs, mask=m, other=0)
    c = tl.load(col_ptr + offs, mask=m, other=0)
    v = tl.load(val_ptr + offs, mask=m, other=0)
    old = tl.atomic_add(cur_ptr + r, -1, mask=m)
    end = tl.load(bptr + r + 1, mask=m, other=0)
    slot = end - old
    tl.store(ocol_ptr + slot, c, mask=m)
    tl.store(oval_ptr + slot, v, mask=m)


@triton.jit
def _sspm_accum(
    arow_ptr, acol_ptr, aval_ptr, bptr, bcol, bval, cdense, vdense, NN, BB: tl.constexpr
):
    # One program per A-nnz entry (i, j, v); accumulate products into the
    # dense (M x N) count/value buffers for every B entry in row j.
    pid = tl.program_id(0)
    i = tl.load(arow_ptr + pid)
    j = tl.load(acol_ptr + pid)
    v = tl.load(aval_ptr + pid)
    start = tl.load(bptr + j)
    end = tl.load(bptr + j + 1)
    acc_t = vdense.dtype.element_ty
    for s in range(start, end, BB):
        offs = s + tl.arange(0, BB)
        m = offs < end
        k = tl.load(bcol + offs, mask=m, other=0)
        w = tl.load(bval + offs, mask=m, other=0)
        addr = i * NN + k
        tl.atomic_add(cdense + addr, 1, mask=m)
        tl.atomic_add(vdense + addr, v.to(acc_t) * w.to(acc_t), mask=m)


@triton.jit
def _sspm_rowcount(cdense, ocnt_ptr, NN, BLOCK: tl.constexpr):
    # One program per output row: count distinct columns present.
    i = tl.program_id(0)
    base = i.to(tl.int64) * NN
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for s in range(0, NN, BLOCK):
        offs = s + tl.arange(0, BLOCK)
        m = offs < NN
        c = tl.load(cdense + base + offs, mask=m, other=0)
        acc += (c > 0).to(tl.int32)
    tl.store(ocnt_ptr + i, tl.sum(acc, axis=0))


@triton.jit
def _sspm_emit(
    cdense, vdense, orow_ptr, orow_out, ocol_out, oval_out, NN, BLOCK: tl.constexpr
):
    # One program per output row: emit (row, col, val) in ascending column
    # order by scanning the dense buffers; structural zeros are preserved.
    i = tl.program_id(0)
    pos = tl.load(orow_ptr + i)
    base = i.to(tl.int64) * NN
    for s in range(0, NN, BLOCK):
        offs = s + tl.arange(0, BLOCK)
        m = offs < NN
        c = tl.load(cdense + base + offs, mask=m, other=0)
        act = m & (c > 0)
        mi = act.to(tl.int32)
        rank = tl.cumsum(mi, axis=0) - mi
        dst = pos + rank
        v = tl.load(vdense + base + offs, mask=act, other=0)
        tl.store(orow_out + dst, i.to(tl.int64), mask=act)
        tl.store(ocol_out + dst, offs.to(tl.int64), mask=act)
        tl.store(oval_out + dst, v.to(oval_out.dtype.element_ty), mask=act)
        pos = pos + tl.sum(mi, axis=0)


def _sparse_sparse_matmul(self, other):
    if not self.is_sparse or not other.is_sparse:
        raise RuntimeError(
            "_sparse_sparse_matmul: expected sparse tensors for both operands"
        )
    if self.dim() != 2 or other.dim() != 2:
        raise RuntimeError(
            "_sparse_sparse_matmul: expected 2-D sparse tensors, got "
            f"{self.dim()}-D and {other.dim()}-D"
        )
    if self.dense_dim() != 0 or other.dense_dim() != 0:
        raise RuntimeError(
            "_sparse_sparse_matmul: expected scalar sparse values (no dense dimensions)"
        )
    if self.shape[1] != other.shape[0]:
        raise RuntimeError(
            f"mat1 and mat2 shapes cannot be multiplied "
            f"({self.shape[0]}x{self.shape[1]} and "
            f"{other.shape[0]}x{other.shape[1]})"
        )
    if self.dtype != other.dtype:
        raise RuntimeError(
            f"mat1 dtype {self.dtype} does not match mat2 dtype {other.dtype}"
        )
    if self.dtype not in _SPARSE_DTYPES:
        raise NotImplementedError(f'"sparse_matmul" not implemented for {self.dtype!s}')

    a = self.coalesce()
    b = other.coalesce()
    M, K = a.shape
    N = other.shape[1]
    dev = a.device
    dt = a.dtype
    nnzA = a._nnz()
    nnzB = b._nnz()

    # ``a`` / ``b`` are coalesced above, so their index and value tensors are
    # already contiguous — no extra copies on the hot path (review P1-2).
    a_idx = a._indices()
    b_idx = b._indices()
    arow = a_idx[0]
    acol = a_idx[1]
    aval = a._values()
    brow = b_idx[0]
    bcol = b_idx[1]
    bval = b._values()

    MN = M * N
    acc_dtype = torch.float64 if dt == torch.float64 else torch.float32

    # ---- CSR grouping of B (histogram -> scan -> scatter); bcnt doubles as
    # the scatter cursor and self-zeroes, so no separate cursor is needed ----
    bcnt = torch.zeros(K, dtype=torch.int32, device=dev)
    if nnzB > 0:
        _sspm_hist[(triton.cdiv(nnzB, 1024),)](brow, bcnt, nnzB, BLOCK=1024)
    bptr = torch.empty(K + 1, dtype=torch.int32, device=dev)
    _sspm_scan[(1,)](bcnt, bptr, K, BLOCK=1024)
    bcol_csr = torch.empty(nnzB, dtype=torch.int64, device=dev)
    bval_csr = torch.empty(nnzB, dtype=dt, device=dev)
    if nnzB > 0:
        _sspm_scatter[(triton.cdiv(nnzB, 1024),)](
            brow, bcol, bval, bcnt, bptr, bcol_csr, bval_csr, nnzB, BLOCK=1024
        )

    # ---- accumulate products into dense (M x N) count/value buffers ----
    # NOTE (documented per the review): the scratch is dense over M*N — two
    # buffers of 4 bytes * MN each — so inputs whose product of sparse dims is
    # far larger than the output nnz can exceed memory. ATen's SpGEMM uses an
    # O(nnz) hash-based scheme; this kernel trades memory for a simpler
    # exact-count pipeline. Keep (M, N) within available device memory.
    cdense = torch.zeros(MN, dtype=torch.int32, device=dev)
    vdense = torch.zeros(MN, dtype=acc_dtype, device=dev)
    if nnzA > 0 and nnzB > 0:
        _sspm_accum[(nnzA,)](
            arow, acol, aval, bptr, bcol_csr, bval_csr, cdense, vdense, N, BB=256
        )

    # ---- exact per-row counts, then the row-major output offsets ----
    ocnt = torch.empty(M, dtype=torch.int32, device=dev)
    orow_ptr = torch.empty(M + 1, dtype=torch.int32, device=dev)
    if M > 0:
        _sspm_rowcount[(M,)](cdense, ocnt, N, BLOCK=1024)
        _sspm_scan[(1,)](ocnt, orow_ptr, M, BLOCK=1024)

    nnz = int(orow_ptr[M].item()) if M > 0 else 0
    orows = torch.empty(nnz, dtype=torch.int64, device=dev)
    ocols = torch.empty(nnz, dtype=torch.int64, device=dev)
    ovals = torch.empty(nnz, dtype=dt, device=dev)
    if nnz > 0:
        _sspm_emit[(M,)](cdense, vdense, orow_ptr, orows, ocols, ovals, N, BLOCK=1024)

    indices = torch.empty((2, nnz), dtype=torch.int64, device=dev)
    indices[0] = orows
    indices[1] = ocols
    out = torch.sparse_coo_tensor(indices, ovals, (M, N), device=dev)
    # The emit kernel writes each (row, col) pair once in ascending row-major
    # order, so the returned COO is genuinely coalesced -- mark it so
    # ``is_coalesced()`` matches ATen without a later sort.
    out._coalesced_(True)
    return out


def _sparse_sparse_matmul_out(self, other, *, out):
    result = _sparse_sparse_matmul(self, other)
    if out.layout != torch.sparse_coo:
        raise RuntimeError(
            f"Expected out tensor to have sparse COO layout, but got {out.layout}"
        )
    if out.dtype != result.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {result.dtype}, but got {out.dtype}"
        )
    if out.shape != result.shape:
        raise RuntimeError(
            f"Expected out tensor to have shape {tuple(result.shape)}, but "
            f"got {tuple(out.shape)}"
        )
    return out.copy_(result)
