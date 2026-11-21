import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def spdiags_kernel(
    diagonals_ptr,
    offsets_ptr,
    indices_ptr,
    values_ptr,
    num_diags: tl.constexpr,
    diag_len: tl.constexpr,
    nrows: tl.constexpr,
    ncols: tl.constexpr,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to compute sparse COO indices and values from diagonals.

    Each program processes one block of one diagonal and writes into a
    fixed-stride ``num_diags * diag_len`` scratch layout: entry ``i`` of
    diagonal ``d`` lands at slot ``d * diag_len + i``. Invalid slots (positions
    that fall outside the matrix) get their row index tagged with the sentinel
    ``-1`` so the host side can compact the results with a single ``nonzero``
    instead of a per-diagonal Python loop.

    Coordinates follow the native ``_spdiags`` layout contract: the ``j``-th
    element of a diagonal lands at matrix position ``(j - offset, j)``. Only
    positions whose row and column fall inside the matrix are valid.

    (Upstream additionally guarded ``diag_idx >= num_diags`` with an early
    return; the grid covers exactly ``num_diags * cdiv(diag_len, BLOCK)``
    programs, so the guard is dead and this backend rejects runtime scalar
    branches -- it is dropped.)
    """
    pid = tl.program_id(0)
    diag_idx = pid // tl.cdiv(diag_len, BLOCK_SIZE)
    block_idx = pid % tl.cdiv(diag_len, BLOCK_SIZE)

    # Load the offset without narrowing to int32: valid offsets are int64.
    offset = tl.load(offsets_ptr + diag_idx)

    positions = (block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    inbound = positions < diag_len

    # Native coordinate mapping: diagonal value at position j maps to
    # (row, col) = (j - offset, j). Filter out-of-bounds indices directly.
    row_idx = positions - offset
    col_idx = positions

    valid = inbound & (row_idx >= 0) & (row_idx < nrows) & (col_idx < ncols)

    # Fixed-stride output position within the scratch buffers.
    out_idx = diag_idx * diag_len + positions

    # Load diagonal values.
    diag_base = diag_idx * diag_len
    vals = tl.load(diagonals_ptr + diag_base + positions, mask=valid, other=0.0)

    # Store into the shared (2, numel) index buffer: row = indices[0], col = indices[1].
    # Invalid entries carry a -1 row sentinel so they are dropped during compaction.
    store_row = tl.where(valid, row_idx, -1)
    tl.store(indices_ptr + out_idx, store_row, mask=inbound)
    tl.store(indices_ptr + numel + out_idx, col_idx, mask=inbound)
    tl.store(values_ptr + out_idx, vals, mask=inbound)


def _to_layout(result_coo, layout):
    if layout == torch.sparse_csr:
        return result_coo.to_sparse_csr()
    if layout == torch.sparse_csc:
        return result_coo.to_sparse_csc()
    return result_coo


def spdiags(diagonals, offsets, shape, layout=None):
    """torch.sparse.spdiags: build a sparse diagonal matrix from diagonals.

    Diagonal entry ``j`` of offset ``k`` lands at ``(j - k, j)``; positions
    outside the matrix are dropped. Offsets must be a long tensor without
    duplicates (aten contract; same error messages as torch.sparse.spdiags).

    Note: torch.sparse.spdiags itself returns a corrupt tensor (heap overflow
    on ``to_dense``) when ``diag_len > ncols`` leaves a kept position with
    ``col >= ncols`` (e.g. diagonals of length 3 into a (3, 2) matrix); this
    port keeps the mathematically correct ``col < ncols`` filter.
    """
    logger.debug("GEMS SPDIAGS")

    # Promote 1-D diagonals and scalar offsets to the native 2-D/1-D forms.
    diagonals_2d = diagonals.unsqueeze(0) if diagonals.dim() == 1 else diagonals
    offsets_1d = offsets.unsqueeze(0) if offsets.dim() == 0 else offsets

    if diagonals_2d.dim() != 2:
        raise RuntimeError("Diagonals must be vector or matrix")
    if offsets_1d.dim() != 1:
        raise RuntimeError("Offsets must be scalar or vector")
    if len(shape) != 2:
        raise RuntimeError("Output shape must be 2d")

    if layout is not None and layout not in (
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
    ):
        raise RuntimeError(
            "Only output layouts (Sparse, SparseCsc, SparseCsr) are supported, "
            f"got {layout}"
        )

    if offsets_1d.dtype != torch.int64:
        raise RuntimeError(
            f"Offset Tensor must have dtype Long but got {offsets_1d.dtype}"
        )

    num_diags = diagonals_2d.shape[0]
    diag_len = diagonals_2d.shape[1]
    nrows, ncols = shape

    if offsets_1d.shape[0] != num_diags:
        raise RuntimeError(
            f"Number of diagonals ({num_diags}) does not match "
            f"the number of offsets ({offsets_1d.shape[0]})"
        )

    # Duplicate detection: torch.unique returns the deduplicated values, so a
    # shrink in element count means the offsets contained duplicates. An empty
    # offset tensor trivially has no duplicates.
    if offsets_1d.numel() > 0:
        unique_offsets = torch.unique(offsets_1d, sorted=False)
        if offsets_1d.numel() != unique_offsets.numel():
            raise RuntimeError("Offset tensor contains duplicate values")

    # Handle empty case
    if num_diags == 0 or diag_len == 0 or nrows == 0 or ncols == 0:
        indices = torch.empty((2, 0), dtype=torch.int64, device=diagonals.device)
        values = torch.empty((0,), dtype=diagonals.dtype, device=diagonals.device)
        result = torch.sparse_coo_tensor(
            indices, values, size=shape, dtype=diagonals.dtype, device=diagonals.device
        )
        return _to_layout(result, layout)

    # Scratch buffers use a fixed ``num_diags * diag_len`` stride so the kernel
    # can write each entry to a deterministic slot without any host-side nnz
    # bookkeeping. The grid covers every slot exactly once, so the kernel fully
    # initializes both buffers: valid entries get their real (row, col) while
    # unused slots receive the -1 row sentinel used by the compaction below.
    numel = num_diags * diag_len
    indices_buffer = torch.empty((2, numel), dtype=torch.int64, device=diagonals.device)
    values_buffer = torch.empty(
        (numel,), dtype=diagonals.dtype, device=diagonals.device
    )

    # Launch kernel
    BLOCK_SIZE = 256
    grid = (num_diags * triton.cdiv(diag_len, BLOCK_SIZE),)

    diagonals_contig = diagonals_2d.contiguous()
    offsets_contig = offsets_1d.contiguous()

    spdiags_kernel[grid](
        diagonals_contig,
        offsets_contig,
        indices_buffer,
        values_buffer,
        num_diags,
        diag_len,
        nrows,
        ncols,
        numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Compact valid entries (row sentinel -1 marks dropped slots).
    keep = torch.nonzero(indices_buffer[0] >= 0).reshape(-1)
    indices = indices_buffer.index_select(1, keep)
    values = values_buffer.index_select(0, keep)

    result = torch.sparse_coo_tensor(
        indices, values, size=shape, dtype=diagonals.dtype, device=diagonals.device
    )
    return _to_layout(result, layout)
