import torch
import triton
import triton.language as tl

_MAX_HERMITE_DEGREE = 9


@triton.jit
def _hermite_Hn(x, n_int, COMPUTE_IN_FP64: tl.constexpr):
    xf = x.to(tl.float64) if COMPUTE_IN_FP64 else x.to(tl.float32)
    h0 = xf * 0.0 + 1.0
    h1 = 2.0 * xf
    result = tl.where(n_int == 1, h1, h0)
    prev = h0
    cur = h1

    # Keep the upstream-supported degree range static for RISC-V lowering.
    for k in range(1, 9):
        nxt = 2.0 * xf * cur - 2.0 * k * prev
        result = tl.where(n_int == k + 1, nxt, result)
        prev = cur
        cur = nxt
    return result


@triton.jit
def hermite_polynomial_h_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    n_elements,
    N_IS_TENSOR: tl.constexpr,
    COMPUTE_IN_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    n = tl.load(n_ptr + offsets, mask=mask).to(tl.int32) if N_IS_TENSOR else n_ptr
    out = _hermite_Hn(x, n, COMPUTE_IN_FP64).to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask)


def special_hermite_polynomial_h_tensor_tensor(x, n):
    return special_hermite_polynomial_h(x, n)


def hermite_polynomial_h_func_tensor_scalar(x, n):
    return special_hermite_polynomial_h(x, n)


def special_hermite_polynomial_h(x, n):
    if not isinstance(x, torch.Tensor):
        raise ValueError("First argument must be a tensor")

    if isinstance(n, torch.Tensor):
        if torch.any((n < 0) | (n > _MAX_HERMITE_DEGREE)).item():
            raise ValueError("special_hermite_polynomial_h only supports n in [0, 9]")
        x_in, n_in = torch.broadcast_tensors(x, n)
    else:
        n = int(n)
        if n < 0 or n > _MAX_HERMITE_DEGREE:
            raise ValueError(
                f"special_hermite_polynomial_h only supports n in [0, 9], got n={n}"
            )
        x_in = x
        n_in = n

    x_c = x_in.contiguous()
    out = torch.empty_like(x_c)
    n_elements = x_c.numel()
    if n_elements == 0:
        return out.view_as(x_in)

    grid = (triton.cdiv(n_elements, 1024),)
    if isinstance(n_in, torch.Tensor):
        hermite_polynomial_h_kernel[grid](
            x_c,
            n_in.contiguous(),
            out,
            n_elements,
            True,
            x.dtype == torch.float64,
            BLOCK_SIZE=1024,
        )
    else:
        hermite_polynomial_h_kernel[grid](
            x_c,
            n_in,
            out,
            n_elements,
            False,
            x.dtype == torch.float64,
            BLOCK_SIZE=1024,
        )
    return out.view_as(x_in)
