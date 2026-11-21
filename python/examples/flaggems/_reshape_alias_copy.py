import torch
import triton
import triton.language as tl

# _reshape_alias_copy: aten::_reshape_alias as a value copy. The view
# described by size/stride (self's storage offset reused) is materialized
# into fresh contiguous storage; size/stride are applied verbatim without
# numel validation, exactly like aten.

_BLOCK_SIZE = 1024
_MAX_RANK = 8


@triton.jit
def _reshape_alias_copy_contig_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    offsets = offsets.to(tl.int64)
    values = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, values, mask=mask)


@triton.jit
def _reshape_alias_copy_strided_kernel(
    src_ptr,
    dst_ptr,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    n_elements,
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    offsets = offsets.to(tl.int64)
    src_strides = [s0, s1, s2, s3, s4, s5, s6, s7]
    dims = [d0, d1, d2, d3, d4, d5, d6, d7]
    rem = offsets
    src_offsets = tl.zeros_like(offsets)
    for j in tl.static_range(RANK - 1, -1, -1):
        idx = rem % dims[j]
        rem = rem // dims[j]
        src_offsets += idx * src_strides[j]
    values = tl.load(src_ptr + src_offsets, mask=mask)
    tl.store(dst_ptr + offsets, values, mask=mask)


def _reshape_alias_copy(self: torch.Tensor, size, stride):
    """Contiguous copy of ``self`` viewed as (size, stride) at self's storage offset."""
    sizes = [int(s) for s in size]
    strides = [int(s) for s in stride]
    out = torch.empty(sizes, dtype=self.dtype, device=self.device)
    n_elements = out.numel()
    if n_elements == 0:
        return out
    view = self.as_strided(sizes, strides, self.storage_offset())
    if view.is_contiguous():
        grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
        _reshape_alias_copy_contig_kernel[grid](
            view,
            out,
            n_elements,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
        return out
    rank = view.dim()
    if rank > _MAX_RANK:
        raise RuntimeError(
            f"_reshape_alias_copy supports at most {_MAX_RANK} dimensions, got {rank}"
        )
    src_strides = list(view.stride()) + [0] * (_MAX_RANK - rank)
    dims = list(view.shape) + [1] * (_MAX_RANK - rank)
    grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
    _reshape_alias_copy_strided_kernel[grid](
        view,
        out,
        *src_strides,
        *dims,
        n_elements,
        RANK=rank,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return out
