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
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if x.dtype != tl.float64:
        x = x.to(tl.float32)
    n = tl.load(n_ptr + offsets, mask=mask, other=0) if N_IS_TENSOR else n_ptr
    n = tl.cast(n, tl.int64)
    n = tl.where(mask, n, 0)
    # Keep the source trigonometric formula where acos is defined.
    outside = tl.abs(x) > 1.0
    acos_x = tl.math.acos(tl.where(outside, 0.0, x))
    nf = n.to(x.dtype)
    out = tl.cos((nf + 0.5) * acos_x) / tl.cos(acos_x * 0.5)
    boundary = tl.where(n % 2 == 0, 2.0 * nf + 1.0, -2.0 * nf - 1.0)
    out = tl.where(x == -1.0, boundary, out)

    # Outside [-1, 1], the polynomial is real but acos is not.
    previous = tl.full(x.shape, 1.0, x.dtype)
    current = 2.0 * x - 1.0
    out = tl.where(outside & (n == 1), current, out)
    for degree in range(2, tl.max(tl.where(outside, n, 0), 0) + 1):
        following = 2.0 * x * current - previous
        out = tl.where(outside & (n == degree), following, out)
        previous = current
        current = following
    out = tl.where(n < 0, 0.0, tl.where(n == 0, 1.0, out))
    tl.store(out_ptr + offsets, out, mask=mask)


def special_chebyshev_polynomial_v(x, n):
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
        chebyshev_polynomial_v_kernel[grid](
            x_c, n.contiguous(), out, n_elements, True, BLOCK_SIZE=1024
        )
    else:
        chebyshev_polynomial_v_kernel[grid](
            x_c, n, out, n_elements, False, BLOCK_SIZE=1024
        )
    return out
