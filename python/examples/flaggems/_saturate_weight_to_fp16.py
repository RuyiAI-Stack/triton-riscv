import torch
import triton
import triton.language as tl


@triton.jit
def _clamp_kernel(x_ptr, out_ptr, n, lo, hi, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = tl.minimum(tl.maximum(x, lo), hi)
    tl.store(out_ptr + offs, y, mask=mask)


def _saturate_weight_to_fp16(A):
    flat = A.contiguous().view(-1).to(torch.float32)
    out = torch.empty_like(flat)
    n = flat.numel()
    if n:
        block = 256
        _clamp_kernel[(triton.cdiv(n, block),)](
            flat, out, n, -65504.0, 65504.0, BLOCK=block
        )
    return out.view(A.shape).to(A.dtype)


__all__ = ["_saturate_weight_to_fp16"]
