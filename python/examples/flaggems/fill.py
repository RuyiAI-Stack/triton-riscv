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


def _fill_scalar_impl(out, value):
    n_elements = out.numel()
    if n_elements == 0:
        return out

    kernel_out = out
    kernel_value = value
    if out.dtype == torch.bool:
        # PyTorch bool tensors use byte storage.  Passing ptr<i1> through
        # Triton creates a pointer bitcast unsupported by the CPU lowering.
        kernel_out = out.view(torch.uint8)
        kernel_value = int(bool(value))

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fill_scalar_kernel[grid](
        kernel_out, kernel_value, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def fill_scalar(input, value):
    out = torch.empty_like(input)
    return _fill_scalar_impl(out, value)


def fill_scalar_out(input, value, *, out=None):
    if out is None:
        return fill_scalar(input, value)

    return _fill_scalar_impl(out, value)


def fill_tensor(input, value):
    if value.ndim != 0:
        raise RuntimeError(
            "fill_ only supports 0-dimension value tensor "
            f"but got tensor with {value.ndim} dimensions."
    )
    out = torch.empty_like(input)
    return _fill_scalar_impl(out, value.item())


def fill_tensor_out(input, value, *, out=None):
    if out is None:
        return fill_tensor(input, value)
    if value.ndim != 0:
        raise RuntimeError(
            "fill_ only supports 0-dimension value tensor "
            f"but got tensor with {value.ndim} dimensions."
        )

    return _fill_scalar_impl(out, value.item())


def fill_tensor_(self, value):
    if value.ndim != 0:
        raise RuntimeError(
            "fill_ only supports 0-dimension value tensor "
            f"but got tensor with {value.ndim} dimensions."
        )

    return _fill_scalar_impl(self, value.item())


def fill_scalar_(self, value):
    return _fill_scalar_impl(self, value)


fill = fill_tensor
fill_ = fill_tensor_
