import torch
import triton
import triton.language as tl


@triton.jit
def chebyshev_polynomial_v_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    n_elements,
    N_IS_TENSOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    n = (
        tl.load(n_ptr + offsets, mask=mask).to(tl.float32)
        if N_IS_TENSOR
        else tl.cast(n_ptr, tl.float32)
    )
    acos_x = tl.math.acos(x)
    out = tl.cos((n + 0.5) * acos_x) / tl.cos(acos_x * 0.5)
    tl.store(out_ptr + offsets, out, mask=mask)


def special_chebyshev_polynomial_v(x, n):
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    n_elements = x_c.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    if isinstance(n, torch.Tensor):
        chebyshev_polynomial_v_kernel[grid](
            x_c, n.contiguous(), out, n_elements, True, BLOCK_SIZE=1024
        )
    else:
        chebyshev_polynomial_v_kernel[grid](
            x_c, n, out, n_elements, False, BLOCK_SIZE=1024
        )
    return out.view_as(x)
