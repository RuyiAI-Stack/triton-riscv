import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 2048
_SCAN_BLOCK = 1024


@triton.jit
def argwhere_count_kernel(
    inp,
    block_counts,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    inp_vals = tl.load(inp + offset, mask=mask, other=0)
    nonzero_mask = (inp_vals != 0) & mask
    tl.store(block_counts + pid, tl.sum(nonzero_mask.to(tl.int32), axis=0))


@triton.jit
def argwhere_scan_kernel(
    block_counts,
    block_offsets,
    total,
    num_blocks,
    BLOCK: tl.constexpr,
):
    # Single program: tiles a running cumsum over the per-block counts and
    # writes each block's output base offset plus the grand total.
    carry = 0
    for start in range(0, num_blocks, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < num_blocks
        counts = tl.load(block_counts + idx, mask=mask, other=0)
        tl.store(
            block_offsets + idx,
            tl.cumsum(counts, axis=0) - counts + carry,
            mask=mask,
        )
        carry += tl.sum(counts, axis=0)
    tl.store(total, carry)


@triton.jit
def argwhere_scatter_kernel(
    inp,
    block_offsets,
    out,
    n_elements,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program re-derives its chunk's non-zero positions with an in-block
    # cumsum over the block's base offset, then decomposes the flat index
    # into coordinates. All index math stays in int32.
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp_vals = tl.load(inp + offset, mask=mask, other=0)
    nonzero_mask = (inp_vals != 0) & mask
    nzi = nonzero_mask.to(tl.int32)

    out_offset = tl.cumsum(nzi, axis=0) - nzi
    out_offset += tl.load(block_offsets + pid)

    row_ptrs = out + out_offset * ndim
    idx = offset
    for i in tl.static_range(3):
        dim = ndim - 1 - i
        # static_range visits all three slots; only dims that exist write.
        if dim >= 0:
            dim_size = D0
            if dim == 1:
                dim_size = D1
            elif dim == 2:
                dim_size = D2
            coord = idx % dim_size
            tl.store(row_ptrs + (ndim - 1 - i), coord.to(tl.int64), mask=nonzero_mask)
            idx = idx // dim_size


def argwhere(input: torch.Tensor):
    if input.ndim > 3:
        return torch.argwhere(input)

    x = input.contiguous().reshape(-1)
    n_elements = x.numel()
    num_blocks = triton.cdiv(n_elements, _BLOCK_SIZE)

    block_counts = torch.empty((num_blocks,), dtype=torch.int32, device=x.device)
    block_offsets = torch.empty((num_blocks,), dtype=torch.int32, device=x.device)
    total = torch.empty((1,), dtype=torch.int32, device=x.device)

    argwhere_count_kernel[(num_blocks,)](
        x, block_counts, n_elements, BLOCK_SIZE=_BLOCK_SIZE
    )
    argwhere_scan_kernel[(1,)](
        block_counts, block_offsets, total, num_blocks, BLOCK=_SCAN_BLOCK
    )

    k = int(total.item())
    out = torch.empty((k, input.ndim), dtype=torch.int64, device=x.device)
    if k == 0:
        return out

    d = input.shape
    argwhere_scatter_kernel[(num_blocks,)](
        x,
        block_offsets,
        out,
        n_elements,
        D0=d[0] if input.ndim >= 1 else 1,
        D1=d[1] if input.ndim >= 2 else 1,
        D2=d[2] if input.ndim >= 3 else 1,
        ndim=input.ndim,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return out
