import torch
import triton
import triton.language as tl


@triton.jit
def acos_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    # Minimax-style approximation on [0, 1], mirrored for negative inputs.
    # Its maximum absolute error is below 7e-5 and it only uses arithmetic and
    # sqrt operations already supported by the Triton-RISCV lowering pipeline.
    negative = x < 0.0
    ax = tl.abs(x)
    y = -0.0187293
    y = y * ax + 0.0742610
    y = y * ax - 0.2121144
    y = y * ax + 1.5707288
    y = y * tl.sqrt(1.0 - ax)
    y = tl.where(negative, 3.141592653589793 - y, y)
    tl.store(y_ptr + offsets, y, mask=mask)


def acos(x):
    x_c = x.contiguous()
    n_elements = x_c.numel()

    out_dtype = x_c.dtype
    if not out_dtype.is_floating_point:
        out_dtype = torch.float32

    y = torch.empty(x_c.shape, dtype=out_dtype, device=x_c.device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    acos_kernel[grid](x_c, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y.view_as(x)
