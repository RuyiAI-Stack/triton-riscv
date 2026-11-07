import torch
import triton
import triton.language as tl


@triton.jit
def arcsin_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.math.asin(x.to(tl.float32))
    tl.store(out_ptr + offsets, y, mask=mask)


def arcsin(x, *, out=None):
    x_c = x.contiguous()
    result = torch.empty_like(
        x_c, dtype=x.dtype if x.dtype.is_floating_point else torch.float32
    )
    n_elements = x_c.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    arcsin_kernel[grid](x_c, result, n_elements, BLOCK_SIZE=1024)
    result = result.view_as(x)
    if out is None:
        return result
    out.copy_(result)
    return out


def arcsin_(x):
    result = arcsin(x)
    x.copy_(result)
    return x


def arcsin_out(x, *, out=None):
    return arcsin(x, out=out)
