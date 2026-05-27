import torch
import triton
import triton.language as tl


@triton.jit
def log_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    y = tl.minimum(x_fp32, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(x_fp32)))
    tl.store(out_ptr + offsets, y, mask=mask)


def log_sigmoid(x):
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    n_elements = x_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    log_sigmoid_kernel[grid](
        x_c,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.view_as(x)
