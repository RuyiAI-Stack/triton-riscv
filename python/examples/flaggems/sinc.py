import torch
import triton
import triton.language as tl


@triton.jit
def sinc_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # sinc(x) = sin(pi * x) / (pi * x) with sinc(0) = 1; computed in fp32
    # because the division is transcendental-precision sensitive.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    x32 = x.to(tl.float32)
    px = x32 * 3.141592653589793
    y = tl.where(x32 == 0.0, 1.0, tl.sin(px) / px).to(x.dtype)

    tl.store(out_ptr + offsets, y, mask=mask)


def sinc(x: torch.Tensor):
    if x.dtype == torch.float64:
        return torch.sinc(x)
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

    BLOCK_SIZE = 1024

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    sinc_kernel[grid](x_c, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if not out.is_contiguous():
        out.copy_(out_c)

    return out
