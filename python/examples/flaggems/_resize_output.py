from typing import List

import torch
import triton
import triton.language as tl

# _resize_output: aten::_resize_output -- out-of-place resize that copies
# the first min(old, new) flat elements into a fresh tensor on the target
# device (a same-shape/same-device request is a pure reshape view, exactly
# like the upstream no-op branch). The in-place variant delegates to
# Tensor.resize_ (upstream behavior). The flat copy is a Triton kernel.


@triton.jit
def _resize_copy_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def _resize_output(inp: torch.Tensor, size: List[int], device: torch.device):
    """Fresh tensor of ``size`` on ``device`` holding the first min(old, new) elements."""
    old_numel = inp.numel()
    new_numel = 1
    for s in size:
        new_numel *= s
    if new_numel == old_numel and inp.device == device:
        return inp.reshape(size)
    out = torch.empty(size, dtype=inp.dtype, device=device)
    copy_numel = min(old_numel, new_numel)
    if copy_numel > 0:
        inp_flat = inp.reshape(-1)[:copy_numel]
        out_flat = out.reshape(-1)[:copy_numel]
        grid = (triton.cdiv(copy_numel, 1024),)
        _resize_copy_kernel[grid](inp_flat, out_flat, copy_numel, BLOCK=1024)
    return out


def _resize_output_(inp: torch.Tensor, size: List[int], device: torch.device):
    """In-place resize via ``Tensor.resize_``; cross-device round-trips a temp copy."""
    old_numel = inp.numel()
    new_numel = 1
    for s in size:
        new_numel *= s
    if inp.device == device:
        inp.resize_(size)
        return inp
    out = torch.empty(size, dtype=inp.dtype, device=device)
    copy_numel = min(old_numel, new_numel)
    if copy_numel > 0:
        inp_flat = inp.reshape(-1)[:copy_numel]
        out_flat = out.reshape(-1)[:copy_numel]
        grid = (triton.cdiv(copy_numel, 1024),)
        _resize_copy_kernel[grid](inp_flat, out_flat, copy_numel, BLOCK=1024)
    inp.resize_(size)
    inp.copy_(out)
    return inp
