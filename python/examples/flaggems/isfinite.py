import torch
import triton
import triton.language as tl


@triton.jit
def isfinite_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    xf = x.to(tl.float32)
    res = (xf == xf) & (tl.abs(xf) < float("inf"))
    tl.store(out_ptr + offsets, res.to(tl.uint8), mask=mask)


def isfinite(A):
    assert isinstance(A, torch.Tensor)
    if A.is_floating_point():
        n_elements = A.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        out = torch.empty_like(A, dtype=torch.uint8)
        A_c = A.contiguous()
        isfinite_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out.to(torch.bool)
    else:
        return torch.full(A.shape, True, dtype=torch.bool, device=A.device)
