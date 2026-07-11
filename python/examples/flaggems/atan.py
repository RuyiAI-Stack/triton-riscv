import torch
import triton
import triton.language as tl


@triton.jit
def _atan_approx(x):
    ax = tl.abs(x)
    reciprocal = ax > 1.0
    z = tl.where(reciprocal, 1.0 / ax, ax)
    reduce = z > 0.4142135623730950
    t = tl.where(reduce, (z - 1.0) / (z + 1.0), z)
    t2 = t * t
    poly = -1.0 / 13.0
    poly = 1.0 / 11.0 + t2 * poly
    poly = -1.0 / 9.0 + t2 * poly
    poly = 1.0 / 7.0 + t2 * poly
    poly = -1.0 / 5.0 + t2 * poly
    poly = 1.0 / 3.0 + t2 * poly
    result = t - t * t2 * poly
    result = tl.where(reduce, 0.7853981633974483 + result, result)
    result = tl.where(reciprocal, 1.5707963267948966 - result, result)
    return tl.where(x < 0.0, -result, result)


@triton.jit
def atan_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = _atan_approx(x)

    tl.store(out_ptr + offsets, y, mask=mask)


def _atan_internal(x: torch.Tensor, out: torch.Tensor):
    n_elements = x.numel()
    if n_elements == 0:
        return out

    x_c = x.contiguous()
    out_c = out if out.is_contiguous() else out.contiguous()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    atan_kernel[grid](x_c, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if not out.is_contiguous():
        out.copy_(out_c)

    return out


def atan(x: torch.Tensor):
    if not x.dtype.is_floating_point:
        result_dtype = torch.float32
        x = x.to(torch.float32)
    else:
        result_dtype = x.dtype

    out = torch.empty_like(x, dtype=result_dtype)
    return _atan_internal(x, out)


def atan_(x: torch.Tensor):
    if not x.dtype.is_floating_point:
        return torch.ops.aten.atan_(x)

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    if x.is_contiguous():
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        atan_kernel[grid](x, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return x
    else:
        y = x.contiguous()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        atan_kernel[grid](y, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        x.copy_(y)
        return x
