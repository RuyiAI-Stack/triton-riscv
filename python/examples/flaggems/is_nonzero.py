import torch
import triton
import triton.language as tl


@triton.jit
def _nonzero_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, x != 0, mask=mask)


def is_nonzero(A):
    if A.numel() != 1:
        raise RuntimeError("bool value of Tensor with more than one value is ambiguous")
    buf = torch.empty(1, dtype=torch.bool, device=A.device)
    _nonzero_kernel[(1,)](A.contiguous().view(-1), buf, 1, BLOCK=1)
    return bool(buf.item())
