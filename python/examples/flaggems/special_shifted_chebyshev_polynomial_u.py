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
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if x.dtype != tl.float64:
        x = x.to(tl.float32)
    n = tl.load(n_ptr + offsets, mask=mask, other=0) if N_IS_TENSOR else n_ptr
    n = tl.cast(n, tl.int64)
    n = tl.where(mask, n, 0)
    out = shifted_chebyshev_polynomial_u_kernel_scalar_n(x, n)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def shifted_chebyshev_polynomial_u_kernel_scalar_n(x, n):
    original_dtype = x.dtype
    if x.dtype != tl.float64:
        x = x.to(tl.float32)
    shifted = 2.0 * x - 1.0
    n = tl.full(x.shape, 0, tl.int64) + tl.cast(n, tl.int64)
    # Preserve the source formula and its endpoint limits on [0, 1].
    outside = tl.abs(shifted) > 1.0
    theta = tl.math.acos(tl.where(outside, 0.0, shifted))
    denom = tl.sin(theta)
    nf = n.to(x.dtype)
    regular = tl.sin((nf + 1.0) * theta) / denom
    boundary = tl.where(
        shifted < 0.0, tl.where(n % 2 != 0, -1.0 - nf, nf + 1.0), nf + 1.0
    )
    out = tl.where(tl.abs(denom) < 1.0e-6, boundary, regular)

    # Clamping outside [0, 1] changes the polynomial; use its recurrence there.
    previous = tl.full(x.shape, 1.0, x.dtype)
    current = 2.0 * shifted
    out = tl.where(outside & (n == 1), current, out)
    for degree in range(2, tl.max(tl.where(outside, n, 0), 0) + 1):
        following = 2.0 * shifted * current - previous
        out = tl.where(outside & (n == degree), following, out)
        previous = current
        current = following
    out = tl.where(n < 0, 0.0, tl.where(n == 0, 1.0, out))
    return out.to(original_dtype)


def special_shifted_chebyshev_polynomial_u(x, n):
    if isinstance(n, torch.Tensor):
        dtype = torch.result_type(x, n)
        x, n = torch.broadcast_tensors(x, n)
        x = x.to(dtype)
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    n_elements = x_c.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    if isinstance(n, torch.Tensor):
        shifted_chebyshev_polynomial_u_kernel[grid](
            x_c, n.contiguous(), out, n_elements, True, BLOCK_SIZE=1024
        )
    else:
        shifted_chebyshev_polynomial_u_kernel[grid](
            x_c, n, out, n_elements, False, BLOCK_SIZE=1024
        )
    return out


def special_shifted_chebyshev_polynomial_u_(x, n):
    x.copy_(special_shifted_chebyshev_polynomial_u(x, n))
    return x
