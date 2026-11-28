import torch
import triton
import triton.language as tl

# fw_primal_copy: aten::_fw_primal_copy -- a plain value copy of the primal
# (the autograd level is part of the signature but unused, exactly like the
# upstream). Contiguous inputs take a flat streaming kernel; strided ones a
# gather that walks the input's shape/strides from device memory (rank
# capped at 8 by the kernel's static loop).


_MAXD = 8
_BLOCK = 1024


@triton.jit
def _copy_flat_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


@triton.jit
def _copy_strided_kernel(
    src_ptr,
    dst_ptr,
    sizes_ptr,
    strides_ptr,
    ndim,
    numel,
    BLOCK: tl.constexpr,
    MAXD: tl.constexpr,
):
    # Gather into a contiguous destination. The flat logical index is
    # row-major: dim 0 is the MOST significant digit, dim ndim-1 the least.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    rem = offs.to(tl.int64)
    src_off = tl.zeros([BLOCK], dtype=tl.int64)
    for d in tl.static_range(MAXD):
        dd = ndim - 1 - d
        active = dd >= 0
        dd_safe = tl.maximum(dd, 0)
        size_d = tl.load(sizes_ptr + dd_safe)
        stride_d = tl.load(strides_ptr + dd_safe)
        size_safe = tl.where(active, size_d, 1)
        stride_safe = tl.where(active, stride_d, 0)
        idx_d = rem % size_safe
        rem = rem // size_safe
        src_off += idx_d.to(tl.int64) * stride_safe
    vals = tl.load(src_ptr + src_off, mask=mask)
    tl.store(dst_ptr + offs, vals, mask=mask)


def _fw_primal_copy(self: torch.Tensor, level: int) -> torch.Tensor:
    """Fresh copy of ``self`` (``level`` is accepted but unused, as upstream)."""
    if not self.is_contiguous() and self.dim() > _MAXD:
        return self.contiguous()
    numel = self.numel()
    if self.is_contiguous():
        out = torch.empty_like(self)
        if numel == 0:
            return out
        grid = (triton.cdiv(numel, _BLOCK),)
        _copy_flat_kernel[grid](self, out, numel, BLOCK=_BLOCK)
        return out
    out = torch.empty_like(self, memory_format=torch.contiguous_format)
    if numel == 0:
        return out
    sizes = torch.tensor(list(self.shape), dtype=torch.int64, device=self.device)
    strides = torch.tensor(list(self.stride()), dtype=torch.int64, device=self.device)
    ndim = len(self.shape)
    grid = (triton.cdiv(numel, _BLOCK),)
    _copy_strided_kernel[grid](
        self,
        out,
        sizes,
        strides,
        ndim,
        numel,
        BLOCK=_BLOCK,
        MAXD=_MAXD,
    )
    return out


def _fw_primal_copy_out(
    self: torch.Tensor, level: int, *, out: torch.Tensor
) -> torch.Tensor:
    """Copy ``self`` into ``out`` (resized like aten; rank > 8 via native copy_)."""
    if out.dtype != self.dtype:
        raise RuntimeError("_fw_primal_copy.out: dtype of input and output must match.")
    if out.device != self.device:
        raise RuntimeError(
            "_fw_primal_copy.out: input and output must be on the same device."
        )
    if out.shape != self.shape:
        out.resize_(self.shape)
    if self.dim() > _MAXD:
        out.copy_(self)
        return out
    numel = self.numel()
    if numel == 0:
        return out
    if self.is_contiguous() and out.is_contiguous():
        grid = (triton.cdiv(numel, _BLOCK),)
        _copy_flat_kernel[grid](self, out, numel, BLOCK=_BLOCK)
        return out
    if self.is_contiguous():
        src = self
    else:
        src = self.contiguous()
    grid = (triton.cdiv(numel, _BLOCK),)
    _copy_flat_kernel[grid](src, out.reshape(-1), numel, BLOCK=_BLOCK)
    return out
