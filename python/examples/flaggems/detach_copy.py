import torch
import triton
import triton.language as tl


@triton.jit
def _copy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


def _copy_values(x):
    flat = x.contiguous().view(-1)
    out = torch.empty_like(flat)
    n = flat.numel()
    if n:
        block = 256
        _copy_kernel[(triton.cdiv(n, block),)](flat, out, n, BLOCK=block)
    return out.view(x.shape)


def detach_copy(self):
    return _copy_values(self)


__all__ = ["detach_copy"]
