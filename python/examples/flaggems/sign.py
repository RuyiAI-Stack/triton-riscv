import torch
import triton
import triton.language as tl


@triton.jit
def sign_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x > 0, 1, tl.where(x < 0, -1, 0)).to(x.dtype)

    tl.store(out_ptr + offsets, y, mask=mask)


def sign(x: torch.Tensor):
    # sign is dtype-preserving: int in, int out (no float promotion).
    out = torch.empty_like(x)

    n_elements = x.numel()
    if n_elements == 0:
        return out

    x_c = x.contiguous()
    out_c = out if out.is_contiguous() else out.contiguous()

    BLOCK_SIZE = 1024

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    sign_kernel[grid](x_c, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if not out.is_contiguous():
        out.copy_(out_c)

    return out
