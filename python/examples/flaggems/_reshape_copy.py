import torch
import triton
import triton.language as tl

# _reshape_copy: aten::_reshape_copy -- the row-major logical order of the
# input materialized into fresh contiguous storage. A single -1 dimension
# may be inferred; the size is validated against numel. Contiguous inputs
# take a flat copy, strided ones a padded 8-slot gather.

_BLOCK_SIZE = 1024
_MAX_RANK = 8


@triton.jit
def _reshape_copy_contig_kernel(
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
def _reshape_copy_strided_kernel(
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


def _reshape_copy(self: torch.Tensor, size):
    """Contiguous copy of ``self`` reshaped to ``size`` (one -1 may be inferred)."""
    requested = [int(s) for s in size]
    numel = self.numel()
    infer_dim = -1
    known = 1
    for i, s in enumerate(requested):
        if s == -1:
            if infer_dim != -1:
                raise RuntimeError("only one dimension can be inferred")
            infer_dim = i
        elif s < 0:
            raise RuntimeError(f"invalid shape dimension {s}")
        else:
            known *= s
    if infer_dim != -1:
        if known == 0:
            raise RuntimeError(
                f"shape '{requested}' is invalid for input of size {numel}"
            )
        requested[infer_dim] = numel // known
    computed = 1
    for s in requested:
        computed *= s
    if computed != numel:
        raise RuntimeError(f"shape '{requested}' is invalid for input of size {numel}")
    out = torch.empty(requested, dtype=self.dtype, device=self.device)
    if numel == 0:
        return out
    if self.is_contiguous():
        grid = (triton.cdiv(numel, _BLOCK_SIZE),)
        _reshape_copy_contig_kernel[grid](
            self,
            out,
            numel,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
        return out
    rank = self.dim()
    if rank > _MAX_RANK:
        raise RuntimeError(
            f"_reshape_copy supports at most {_MAX_RANK} dimensions, got {rank}"
        )
    src_strides = list(self.stride()) + [0] * (_MAX_RANK - rank)
    dims = list(self.shape) + [1] * (_MAX_RANK - rank)
    grid = (triton.cdiv(numel, _BLOCK_SIZE),)
    _reshape_copy_strided_kernel[grid](
        self,
        out,
        *src_strides,
        *dims,
        numel,
        RANK=rank,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return out
