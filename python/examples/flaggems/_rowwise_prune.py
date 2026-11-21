import torch
import triton
import triton.language as tl

# _rowwise_prune: aten::_rowwise_prune -- keep the rows of a 2-D weight
# whose mask entry is True. The compressed index kernel gives every kept
# row its exclusive prefix count (-1 for pruned rows); a 2-D grid scatter
# kernel copies kept rows into their compressed slots. M beyond one block
# goes through count/prefix/multi three-kernel passes (cumsum-based).

_PRUNED_ROW = tl.constexpr(-1)
_BLOCK_M = 1024


@triton.jit(do_not_specialize=["M"])
def _rowwise_prune_indices_kernel(
    mask_ptr,
    row_indices_ptr,
    M,
    BLOCK_M: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_M)
    valid = offsets < M
    keep = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int1)
    keep_int = keep.to(tl.int32)
    # Exclusive prefix sum: number of kept rows before each row.
    prefix = tl.cumsum(keep_int, axis=0) - keep_int
    row_index = tl.where(keep, prefix, _PRUNED_ROW)
    tl.store(
        row_indices_ptr + offsets,
        row_index.to(row_indices_ptr.dtype.element_ty),
        mask=valid,
    )


@triton.jit(do_not_specialize=["M"])
def _rowwise_prune_block_count_kernel(
    mask_ptr,
    block_counts_ptr,
    M,
    BLOCK_M: tl.constexpr,
):
    block = tl.program_id(0)
    offsets = block * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = offsets < M
    keep = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int1)
    tl.store(block_counts_ptr + block, tl.sum(keep.to(tl.int32), axis=0))


@triton.jit(do_not_specialize=["num_blocks"])
def _rowwise_prune_block_prefix_kernel(
    block_counts_ptr,
    block_bases_ptr,
    num_blocks,
    BLOCK_NB: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_NB)
    valid = offsets < num_blocks
    counts = tl.load(block_counts_ptr + offsets, mask=valid, other=0)
    prefixes = tl.cumsum(counts, axis=0) - counts
    tl.store(block_bases_ptr + offsets, prefixes, mask=valid)


@triton.jit(do_not_specialize=["M"])
def _rowwise_prune_indices_multi_kernel(
    mask_ptr,
    block_bases_ptr,
    row_indices_ptr,
    M,
    BLOCK_M: tl.constexpr,
):
    block = tl.program_id(0)
    offsets = block * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = offsets < M
    keep = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int1)
    keep_int = keep.to(tl.int32)
    prefix_in_block = tl.cumsum(keep_int, axis=0) - keep_int
    base = tl.load(block_bases_ptr + block)
    row_index = tl.where(keep, base + prefix_in_block, _PRUNED_ROW)
    tl.store(
        row_indices_ptr + offsets,
        row_index.to(row_indices_ptr.dtype.element_ty),
        mask=valid,
    )


@triton.jit(do_not_specialize=["K"])
def _rowwise_prune_scatter_kernel(
    weight_ptr,
    row_indices_ptr,
    out_ptr,
    K,
    BLOCK_K: tl.constexpr,
):
    # 2-D grid: axis 0 enumerates weight rows, axis 1 the K tiles.
    row = tl.program_id(0)
    k_tile = tl.program_id(1)
    out_row = tl.load(row_indices_ptr + row)
    if out_row >= 0:
        k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        vals = tl.load(weight_ptr + row * K + k_offsets, mask=k_mask)
        tl.store(out_ptr + out_row * K + k_offsets, vals, mask=k_mask)


def _compute_row_indices(mask, M, compressed_indices_dtype):
    row_indices = torch.empty(M, dtype=compressed_indices_dtype, device=mask.device)
    if M <= _BLOCK_M:
        _rowwise_prune_indices_kernel[(1,)](
            mask, row_indices, M, BLOCK_M=triton.next_power_of_2(M)
        )
    else:
        num_blocks = triton.cdiv(M, _BLOCK_M)
        block_counts = torch.empty(num_blocks, dtype=torch.int32, device=mask.device)
        block_bases = torch.empty(num_blocks, dtype=torch.int32, device=mask.device)
        _rowwise_prune_block_count_kernel[(num_blocks,)](
            mask, block_counts, M, BLOCK_M=_BLOCK_M
        )
        _rowwise_prune_block_prefix_kernel[(1,)](
            block_counts,
            block_bases,
            num_blocks,
            BLOCK_NB=triton.next_power_of_2(num_blocks),
        )
        _rowwise_prune_indices_multi_kernel[(num_blocks,)](
            mask, block_bases, row_indices, M, BLOCK_M=_BLOCK_M
        )
    return row_indices


def _rowwise_prune_impl(weight, mask, compressed_indices_dtype):
    if weight.dim() != 2:
        raise RuntimeError("'weights' should have 2 dimensions.")
    if mask.dtype != torch.bool:
        raise RuntimeError(f"expected scalar type Bool but found {mask.dtype}")
    if mask.numel() != weight.size(0):
        raise RuntimeError(
            "Number of elements in 'mask' should be equivalent to the number of rows "
            "in 'weights'."
        )
    if compressed_indices_dtype not in (torch.int32, torch.int64):
        raise RuntimeError(
            "compressed_indices_dtype should be either int(int32) or long(int64)."
        )
    weight = weight.contiguous()
    mask = mask.contiguous()
    M, K = weight.shape
    row_indices = (
        _compute_row_indices(mask, M, compressed_indices_dtype) if M > 0 else None
    )
    # The compressed indices of kept rows are ``0 .. nnz - 1``, so counting
    # the non-sentinel entries yields the number of kept rows.
    nnz = int((row_indices >= 0).sum().item()) if M > 0 else 0
    out = torch.empty((nnz, K), dtype=weight.dtype, device=weight.device)
    if nnz > 0:
        BLOCK_K = min(max(triton.next_power_of_2(K), 16), 4096)
        num_k_tiles = triton.cdiv(K, BLOCK_K)
        _rowwise_prune_scatter_kernel[(M, num_k_tiles)](
            weight, row_indices, out, K, BLOCK_K=BLOCK_K
        )
    return out, row_indices


def _rowwise_prune(weight, mask, compressed_indices_dtype):
    """Keep the mask-True rows of a 2-D weight; returns (compressed weight, row indices)."""
    return _rowwise_prune_impl(weight, mask, compressed_indices_dtype)
