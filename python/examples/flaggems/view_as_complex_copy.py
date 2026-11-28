import torch
import triton
import triton.language as tl

# view_as_complex_copy: aten::view_as_complex as a value copy. The input's
# last dimension must be size 2 with stride 1; float32 -> complex64 (etc.).
# The kernel writes through the real view of the output; a RANK==2 strided
# specialization covers the common sliced case, a padded 5-slot generic
# loop everything else (rank capped at 6 with the pair dimension).

_BLOCK_SIZE = 1024
_MAX_LEAD_RANK = 5

_COMPLEX_DTYPE = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


@triton.jit
def _view_as_complex_copy_contig_kernel(
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
def _view_as_complex_copy_strided_kernel(
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


def view_as_complex_copy(self: torch.Tensor) -> torch.Tensor:
    """Fresh complex copy of a real ``(..., 2)`` tensor (last dim size 2, stride 1)."""
    if self.dim() == 0:
        raise RuntimeError("Input tensor must have one or more dimensions")
    if self.size(-1) != 2:
        raise RuntimeError("Tensor must have a last dimension of size 2")
    complex_dtype = _COMPLEX_DTYPE.get(self.dtype)
    if complex_dtype is None:
        raise RuntimeError(
            "view_as_complex is only supported for half, float and double "
            "tensors, but got a tensor of scalar type: "
            f"{str(self.dtype).replace('torch.', '').capitalize()}"
        )
    if self.stride(-1) != 1:
        raise RuntimeError("Tensor must have a last dimension with stride 1")

    if self.is_conj():
        self = self.resolve_conj()
    if self.is_neg():
        self = self.resolve_neg()

    leading = tuple(self.shape[:-1])
    out = torch.empty(leading, dtype=complex_dtype, device=self.device)
    n_complex = out.numel()
    if n_complex == 0:
        return out

    out_real = torch.view_as_real(out)
    grid = (triton.cdiv(n_complex, 1024),)
    if self.is_contiguous():
        _view_as_complex_copy_contig_kernel[grid](
            self,
            out_real,
            n_complex,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
    else:
        rank = self.dim() - 1
        if rank > _MAX_LEAD_RANK:
            raise RuntimeError(
                f"view_as_complex_copy supports at most {_MAX_LEAD_RANK + 1} "
                f"dimensions, got {self.dim()}"
            )
        lead_dims = list(self.shape[:-1]) + [1] * (_MAX_LEAD_RANK - rank)
        lead_strides = list(self.stride()[:-1]) + [0] * (_MAX_LEAD_RANK - rank)
        _view_as_complex_copy_strided_kernel[grid](
            self,
            out_real,
            *lead_dims,
            *lead_strides,
            n_complex,
            RANK=rank,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
    return out
