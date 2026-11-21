import torch
import triton
import triton.language as tl

# select_copy: aten::select as a value copy (rank drops by one).
# Two Triton paths: a linear copy when the selected slice is contiguous
# (contiguous input with dim == 0), and a general strided gather otherwise
# (strides/dims of the *slice* padded to 8 slots). Rank is capped at 8.

_BLOCK_SIZE = 1024
_MAX_RANK = 8


@triton.jit
def _select_copy_contig_kernel(
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
def _select_copy_strided_kernel(
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
    src_offset,
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    offsets = offsets.to(tl.int64)

    src_strides = [s0, s1, s2, s3, s4, s5, s6, s7]
    slice_dims = [d0, d1, d2, d3, d4, d5, d6, d7]

    rem = offsets
    src_offsets = tl.full([], src_offset, tl.int64)
    for j in tl.static_range(RANK - 2, -1, -1):
        idx = rem % slice_dims[j]
        rem = rem // slice_dims[j]
        src_offsets += idx * src_strides[j]

    values = tl.load(src_ptr + src_offsets, mask=mask)
    tl.store(dst_ptr + offsets, values, mask=mask)


def select_copy(self: torch.Tensor, dim: int, index: int):
    """Copy of ``self.select(dim, index)``; contiguous, one rank lower, never a view."""
    if self.dim() == 0:
        raise RuntimeError("select() cannot be applied to a 0-dim tensor.")

    dim_size = self.size(dim)
    if index < -dim_size or index >= dim_size:
        raise RuntimeError(
            f"select(): index {index} out of range for tensor of size "
            f"[{', '.join(str(s) for s in self.shape)}] at dimension {dim}"
        )
    dim = dim % self.dim()
    index = index % dim_size

    shape = list(self.shape)
    del shape[dim]
    out = torch.empty(shape, dtype=self.dtype, device=self.device)

    n_elements = out.numel()
    if n_elements == 0:
        return out

    src_offset = index * int(self.stride(dim))

    if self.is_contiguous() and (dim == 0 or n_elements <= 1):
        grid = (triton.cdiv(n_elements, 1024),)
        _select_copy_contig_kernel[grid](
            self.select(dim, index),
            out,
            n_elements,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
        return out

    rank = self.dim()
    if rank > _MAX_RANK:
        raise RuntimeError(
            f"select_copy supports at most {_MAX_RANK} dimensions, got {rank}"
        )
    src_strides = [int(s) for s in self.stride()]
    slice_dims = list(self.shape)
    del src_strides[dim]
    del slice_dims[dim]
    src_strides = src_strides + [0] * (_MAX_RANK - len(src_strides))
    slice_dims = slice_dims + [1] * (_MAX_RANK - len(slice_dims))

    grid = (triton.cdiv(n_elements, 1024),)
    _select_copy_strided_kernel[grid](
        self,
        out,
        *src_strides,
        *slice_dims,
        n_elements,
        src_offset,
        RANK=rank,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return out
