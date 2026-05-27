import torch
import triton
import triton.language as tl


@triton.jit
def abs_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.abs(x)
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _abs_complex_kernel(
    ri_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    base = offsets * 2
    re = tl.load(ri_ptr + base, mask=mask)
    im = tl.load(ri_ptr + base + 1, mask=mask)
    y = tl.sqrt(re * re + im * im)
    tl.store(y_ptr + offsets, y, mask=mask)


def _abs_complex(x):
    ri = torch.view_as_real(x).contiguous()
    n_elements = x.numel()
    out_dtype = x.real.dtype
    out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _abs_complex_kernel[grid](ri, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def abs(x):
    if x.is_complex():
        return _abs_complex(x)
    x_c = x.contiguous()
    n_elements = x_c.numel()
    y = torch.empty_like(x_c)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    abs_kernel[grid](x_c, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y.view_as(x)


def abs_(x):
    if x.is_complex():
        result = _abs_complex(x)
        x.copy_(result)
        return x
    x_c = x if x.is_contiguous() else x.contiguous()
    n_elements = x_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    abs_kernel[grid](x_c, x_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    if not x.is_contiguous():
        x.copy_(x_c)
    return x
