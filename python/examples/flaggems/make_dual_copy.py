import torch
import triton
import triton.language as tl

# make_dual_copy: aten::make_dual_copy -- an independent copy of ``primal``
# carrying forward-AD dual metadata (the metadata itself cannot be produced
# by a device kernel, so the device work here is the copy; ``tangent`` is
# part of the aten signature but contributes no values). Flat streaming
# kernel for contiguous inputs; complex128 falls back to the native
# strided copy (no 128-bit integer view exists for its strides).


@triton.jit
def _copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask, other=0)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def _as_copyable(t):
    return t.detach() if isinstance(t, torch.Tensor) and t.requires_grad else t


def _is_complex128(t):
    return isinstance(t, torch.Tensor) and t.dtype == torch.complex128


def _strided_copy(src, dst):
    # ``dst`` may preserve non-contiguous strides (empty_like), so the
    # gather goes through a flat staging buffer and a native strided copy
    # places it; writing into ``dst.reshape(-1)`` would hit a temporary.
    src_flat = src.contiguous().reshape(-1)
    n = src_flat.numel()
    staging = torch.empty(n, dtype=src.dtype, device=src.device)
    grid = (triton.cdiv(n, 1024),)
    _copy_kernel[grid](src_flat, staging, n, BLOCK=1024)
    dst.copy_(staging.view(dst.shape))


def _flat_copy(src, dst):
    # Triton cannot bind complex pointers; complex inputs copy through
    # their real views (same layout, twice the lanes).
    if src.is_complex():
        src = torch.view_as_real(src)
        dst = torch.view_as_real(dst)
    n = src.numel()
    grid = (triton.cdiv(n, 1024),)
    _copy_kernel[grid](src, dst, n, BLOCK=1024)


def _make_dual_copy(
    primal: torch.Tensor, tangent: torch.Tensor, level: int
) -> torch.Tensor:
    """Copy of ``primal`` (``tangent`` accepted but unused for the values)."""
    out = torch.empty_like(primal)
    n = primal.numel()
    if n == 0:
        return out
    if primal.is_contiguous():
        _flat_copy(primal, out)
    elif _is_complex128(primal):
        out.copy_(primal)
    else:
        _strided_copy(primal, out)
    return out


def _make_dual_copy_out(
    primal: torch.Tensor, tangent: torch.Tensor, level: int, *, out: torch.Tensor
) -> torch.Tensor:
    """Copy of ``primal`` into ``out`` and return ``out``."""
    n = primal.numel()
    if n == 0:
        return out
    if primal.is_contiguous() and out.is_contiguous():
        _flat_copy(primal, out)
        return out
    if _is_complex128(primal):
        out.copy_(primal)
        return out
    _strided_copy(primal, out)
    return out
