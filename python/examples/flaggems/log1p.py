import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def log1p_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    COMPUTE_FP32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Masked lanes load 0.0: log1p(0.0) == 0.0 keeps the math in-domain.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    if COMPUTE_FP32:
        y = libdevice.log1p(x.to(tl.float32)).to(x.dtype)
    else:
        y = libdevice.log1p(x)

    tl.store(out_ptr + offsets, y, mask=mask)


def log1p(x: torch.Tensor):
    if not x.dtype.is_floating_point:
        out = torch.empty_like(x, dtype=torch.float32)
        x = x.to(torch.float32)
    else:
        out = torch.empty_like(x)

    n_elements = x.numel()
    if n_elements == 0:
        return out

    x_c = x.contiguous()
    out_c = out if out.is_contiguous() else out.contiguous()

    use_fp32 = x.dtype in (torch.float16, torch.bfloat16)
    BLOCK_SIZE = 1024

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    log1p_kernel[grid](
        x_c, out_c, n_elements, COMPUTE_FP32=use_fp32, BLOCK_SIZE=BLOCK_SIZE
    )

    if not out.is_contiguous():
        out.copy_(out_c)

    return out
