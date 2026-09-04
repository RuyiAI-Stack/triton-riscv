import torch
import triton
import triton.language as tl


@triton.jit
def asin_kernel(
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
    y = tl.math.asin(x.to(tl.float32))

    tl.store(out_ptr + offsets, y, mask=mask)


def _asin_internal(x: torch.Tensor, out: torch.Tensor):
    n_elements = x.numel()
    if n_elements == 0:
        return out

    x_c = x.contiguous()
    out_c = out if out.is_contiguous() else out.contiguous()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    asin_kernel[grid](x_c, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if not out.is_contiguous():
        out.copy_(out_c)

    return out


def asin(x: torch.Tensor):
    if not x.dtype.is_floating_point:
        result_dtype = torch.float32
        x = x.to(torch.float32)
    else:
        result_dtype = x.dtype

    out = torch.empty_like(x, dtype=result_dtype)
    return _asin_internal(x, out)


def asin_(x: torch.Tensor):
    if not x.dtype.is_floating_point:
        return torch.ops.aten.asin_(x)

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    if x.is_contiguous():
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        asin_kernel[grid](x, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return x
    else:
        y = x.contiguous()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        asin_kernel[grid](y, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        x.copy_(y)
        return x
