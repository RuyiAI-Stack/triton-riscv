import torch
import triton
import triton.language as tl


@triton.jit
def logaddexp_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    m = tl.maximum(x_f32, y_f32)
    delta = x_f32 - y_f32
    res = m + tl.log(1.0 + tl.exp(-tl.abs(delta)))
    tl.store(out_ptr + offsets, res, mask=mask)


def logaddexp(A, B):
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        A, B = torch.broadcast_tensors(A, B)
        A_c = A.contiguous()
        B_c = B.contiguous()
        out = torch.empty_like(A_c)
        n_elements = A_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        logaddexp_kernel[grid](
            A_c, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        return out.view_as(A)
    else:
        return torch.logaddexp(A, B)


def logaddexp_out(A, B, out):
    result = logaddexp(A, B)
    out.copy_(result.view_as(out))
    return out
