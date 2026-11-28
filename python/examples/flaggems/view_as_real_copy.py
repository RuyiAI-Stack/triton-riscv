import torch
import triton
import triton.language as tl

# view_as_real_copy: aten::view_as_real as a value copy. complex64 ->
# float32 (..., 2), complex128 -> float64, complex32 -> float16. The kernel
# reads through the input's real view (conj/neg resolved on the host); a
# RANK==2 strided specialization covers the common sliced case, a padded
# 5-slot generic loop everything else (rank capped at 5).

_BLOCK_SIZE = 1024
_MAX_LEAD_RANK = 5

_REAL_DTYPE = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


@triton.jit
def _view_as_real_copy_contig_kernel(
    src_ptr,
    dst_ptr,
    n_complex,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i < n_complex
    i = i.to(tl.int64)

    lane = tl.arange(0, 2)
    offsets = i[:, None] * 2 + lane[None, :]
    mask2 = mask[:, None]
    values = tl.load(src_ptr + offsets, mask=mask2)
    tl.store(dst_ptr + offsets, values, mask=mask2)


@triton.jit
def _view_as_real_copy_strided_kernel(
    src_ptr,
    dst_ptr,
    d0,
    d1,
    d2,
    d3,
    d4,
    s0,
    s1,
    s2,
    s3,
    s4,
    n_complex,
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i < n_complex
    i = i.to(tl.int64)

    if RANK == 2:
        outer = i // d1
        src_base = (i - outer * d1) * s1 + outer * s0
    else:
        lead_dims = [d0, d1, d2, d3, d4]
        lead_strides = [s0, s1, s2, s3, s4]
        rem = i
        src_base = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
        for j in tl.static_range(RANK - 1, -1, -1):
            idx = rem % lead_dims[j]
            rem = rem // lead_dims[j]
            src_base += idx * lead_strides[j]

    lane = tl.arange(0, 2)
    src_offsets = src_base[:, None] + lane[None, :]
    dst_offsets = i[:, None] * 2 + lane[None, :]
    mask2 = mask[:, None]
    values = tl.load(src_ptr + src_offsets, mask=mask2)
    tl.store(dst_ptr + dst_offsets, values, mask=mask2)


def view_as_real_copy(self: torch.Tensor) -> torch.Tensor:
    """Fresh ``(..., 2)`` real copy of a complex tensor; non-complex dtypes raise."""
    real_dtype = _REAL_DTYPE.get(self.dtype)
    if real_dtype is None:
        raise RuntimeError(
            f"view_as_real is only supported for complex tensors, not for "
            f"{str(self.dtype).replace('torch.', '')} tensors"
        )

    if self.is_conj():
        self = self.resolve_conj()
    if self.is_neg():
        self = self.resolve_neg()

    out = torch.empty(tuple(self.shape) + (2,), dtype=real_dtype, device=self.device)
    n_complex = self.numel()
    if n_complex == 0:
        return out

    src_real = torch.view_as_real(self)
    grid = (triton.cdiv(n_complex, 1024),)
    if src_real.is_contiguous():
        _view_as_real_copy_contig_kernel[grid](
            src_real,
            out,
            n_complex,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
    else:
        rank = self.dim()
        if rank > _MAX_LEAD_RANK:
            raise RuntimeError(
                f"view_as_real_copy supports at most {_MAX_LEAD_RANK} "
                f"dimensions, got {self.dim()}"
            )
        dims = list(src_real.shape[:-1]) + [1] * (_MAX_LEAD_RANK - rank)
        strides = list(src_real.stride()[:-1]) + [0] * (_MAX_LEAD_RANK - rank)
        _view_as_real_copy_strided_kernel[grid](
            src_real,
            out,
            *dims,
            *strides,
            n_complex,
            RANK=rank,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
    return out
