import torch
import triton
import triton.language as tl


@triton.jit
def shifted_chebyshev_polynomial_u_kernel(
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
    shifted = tl.minimum(tl.maximum(x * 2.0 - 1.0, -1.0), 1.0)
    theta = tl.math.acos(shifted)
    denom = tl.sin(theta)
    regular = tl.sin((n + 1.0) * theta) / denom
    n_mod_2 = n - 2.0 * tl.floor(n / 2.0)
    odd = tl.abs(n_mod_2 - 1.0) < 0.5
    boundary = tl.where(shifted < 0.0, tl.where(odd, -1.0 - n, n + 1.0), n + 1.0)
    out = tl.where(tl.abs(denom) < 1.0e-6, boundary, regular)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def shifted_chebyshev_polynomial_u_kernel_scalar_n(x, n):
    shifted = tl.minimum(tl.maximum(x.to(tl.float32) * 2.0 - 1.0, -1.0), 1.0)
    theta = tl.math.acos(shifted)
    denom = tl.sin(theta)
    nf = n.to(tl.float32)
    regular = tl.sin((nf + 1.0) * theta) / denom
    n_mod_2 = nf - 2.0 * tl.floor(nf / 2.0)
    odd = tl.abs(n_mod_2 - 1.0) < 0.5
    boundary = tl.where(shifted < 0.0, tl.where(odd, -1.0 - nf, nf + 1.0), nf + 1.0)
    return tl.where(tl.abs(denom) < 1.0e-6, boundary, regular).to(x.dtype)


def special_shifted_chebyshev_polynomial_u(x, n):
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    n_elements = x_c.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    if isinstance(n, torch.Tensor):
        shifted_chebyshev_polynomial_u_kernel[grid](
            x_c, n.contiguous(), out, n_elements, True, BLOCK_SIZE=1024
        )
    else:
        shifted_chebyshev_polynomial_u_kernel[grid](
            x_c, n, out, n_elements, False, BLOCK_SIZE=1024
        )
    return out.view_as(x)


def special_shifted_chebyshev_polynomial_u_(x, n):
    x.copy_(special_shifted_chebyshev_polynomial_u(x, n))
    return x
