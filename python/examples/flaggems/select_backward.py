import math

import torch
import triton
import triton.language as tl


@triton.jit
def _select_backward_kernel(
    grad_ptr,
    out_ptr,
    inner_size,
    dim_stride,
    index,
):
    output_offset = tl.program_id(0)
    outer = output_offset // dim_stride
    within_outer = output_offset % dim_stride
    dim_coordinate = within_outer // inner_size
    inner = within_outer % inner_size
    grad_value = tl.load(grad_ptr + outer * inner_size + inner)
    tl.store(
        out_ptr + output_offset,
        tl.where(dim_coordinate == index, grad_value, 0.0),
    )


def _launch_select_backward(grad, input_sizes, dim, index, out=None):
    dim = int(dim)
    index = int(index)

    sizes = list(input_sizes)
    ndim = len(sizes)

    if dim < 0:
        dim += ndim

    if dim < 0 or dim >= ndim:
        raise ValueError("invalid dim")

    dim_size = sizes[dim]

    if index < 0 or index >= dim_size:
        raise ValueError("index out of range")

    outer_size = math.prod(sizes[:dim]) if dim > 0 else 1
    inner_size = math.prod(sizes[dim + 1 :]) if dim < ndim - 1 else 1

    grad_view = grad.contiguous().view(outer_size, inner_size)

    if out is None:
        out = torch.zeros(
            sizes,
            dtype=grad.dtype,
            device=grad.device,
        )
    else:
        if tuple(out.shape) != tuple(sizes):
            raise ValueError("out shape mismatch")
        if out.dtype != grad.dtype:
            raise ValueError("dtype mismatch")
        if out.device != grad.device:
            raise ValueError("device mismatch")

        out.zero_()

    dim_stride = dim_size * inner_size

    n_elements = math.prod(sizes)
    _select_backward_kernel[(n_elements,)](
        grad_view,
        out,
        inner_size,
        dim_stride,
        index,
        num_warps=1,
    )

    return out


def select_backward(grad, input_sizes, dim, index, out=None):
    return _launch_select_backward(grad, input_sizes, dim, index, out=out)
