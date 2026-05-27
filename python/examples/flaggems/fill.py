import torch
import triton
import triton.language as tl


@triton.jit
def fill_scalar_kernel(
    out_ptr,
    value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    tl.store(out_ptr + offsets, value, mask=mask)


def fill_scalar(input, value):
    out = torch.empty_like(input)
    n_elements = input.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fill_scalar_kernel[grid](out, value, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def fill_scalar_out(input, value, *, out=None):
    if out is None:
        return fill_scalar(input, value)

    n_elements = out.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fill_scalar_kernel[grid](out, value, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def fill_tensor(input, value):
    if value.ndim != 0:
        raise RuntimeError(
            "fill_ only supports 0-dimension value tensor "
            f"but got tensor with {value.ndim} dimensions."
        )
    out = torch.empty_like(input)
    n_elements = input.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fill_scalar_kernel[grid](
        out, value.item(), n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def fill_tensor_out(input, value, *, out=None):
    if out is None:
        return fill_tensor(input, value)
    if value.ndim != 0:
        raise RuntimeError(
            "fill_ only supports 0-dimension value tensor "
            f"but got tensor with {value.ndim} dimensions."
        )

    n_elements = out.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fill_scalar_kernel[grid](
        out, value.item(), n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def fill_tensor_(self, value):
    if value.ndim != 0:
        raise RuntimeError(
            "fill_ only supports 0-dimension value tensor "
            f"but got tensor with {value.ndim} dimensions."
        )

    n_elements = self.numel()
    if n_elements == 0:
        return self

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fill_scalar_kernel[grid](
        self, value.item(), n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return self


def fill_scalar_(self, value):
    n_elements = self.numel()
    if n_elements == 0:
        return self

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fill_scalar_kernel[grid](self, value, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return self


fill = fill_tensor
fill_ = fill_tensor_
