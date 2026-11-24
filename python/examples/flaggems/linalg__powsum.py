import torch
import triton
import triton.language as tl


@triton.jit
def _powsum_kernel(x_ptr, out_ptr, n, ord, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    tl.store(out_ptr + offs, tl.exp(ord * tl.log(tl.abs(x) + 1.0e-30)), mask=mask)


def linalg__powsum(inp, ord=2, dim=None, keepdim=False, *, dtype=None):
    work = inp if dtype is None else inp.to(dtype)
    flat = work.reshape(-1).contiguous()
    powered = torch.empty_like(flat, dtype=torch.float32)
    n = flat.numel()
    if n:
        block = 256
        _powsum_kernel[(triton.cdiv(n, block),)](
            flat, powered, n, float(ord), BLOCK=block
        )
    powered = powered.view(work.shape)
    if dim is None:
        result = powered.sum()
        return result.view([1] * work.ndim) if keepdim else result
    return powered.sum(dim=dim, keepdim=keepdim)
