from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

# _stack: aten::_stack -- stack equal-shape tensors along a new ``dim``
# with dtype promotion. Batches of (up to) 4 tensors go through one Triton
# kernel (2-D grid: element blocks x tensor-in-batch); complex inputs stack
# through their real views and reinterpret back, matching aten.


@triton.jit
def stack_copy_kernel(
    out_ptr,
    in_ptr,
    dim_size_out,
    dim_prod_post,
    dim_offset,
    total_elements,
    BLOCK_X: tl.constexpr,
):
    # One launch per input tensor. (Upstream batches four tensors behind
    # runtime pointer selects, which this backend cannot lower: selecting
    # between !tt.ptr values aborts codegen.)
    pid_x = tl.program_id(0)
    block_start = pid_x.to(tl.int64) * BLOCK_X
    offsets = tl.arange(0, BLOCK_X).to(tl.int64)
    idx = block_start + offsets
    scalar_zero = offsets * 0
    dim_size_out = dim_size_out + scalar_zero
    dim_prod_post = dim_prod_post + scalar_zero
    dim_offset = dim_offset + scalar_zero
    mask = idx < total_elements
    pre_idx = idx // dim_prod_post
    post_idx = idx % dim_prod_post
    out_idx = (
        pre_idx * dim_size_out * dim_prod_post + dim_offset * dim_prod_post + post_idx
    )
    vals = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + out_idx, vals, mask=mask)


def _stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    """Stack equal-shape ``tensors`` along new ``dim`` (dtype promotion; 4-tensor kernel batches)."""
    if len(tensors) == 0:
        raise RuntimeError("stack expected a non-empty TensorList")
    inp_shapes = [list(_.shape) for _ in tensors]
    inp0_shape = inp_shapes[0]
    # Validate ``dim`` against tensors[0] unconditionally: with a single
    # input the per-entry loop below never runs and an out-of-range dim
    # would be silently accepted instead of matching ATen.
    ndim0 = tensors[0].dim()
    if (dim < -ndim0 - 1) or (dim > ndim0):
        raise IndexError(
            "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                -ndim0 - 1, ndim0, dim
            )
        )
    for i, s in enumerate(inp_shapes[1:]):
        ndim = tensors[i + 1].dim()
        if (dim < -ndim - 1) or (dim > ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -ndim - 1, ndim, dim
                )
            )
        if s != inp0_shape:
            raise RuntimeError(
                f"stack expects each tensor to be equal size, but got {inp0_shape} at entry 0 and {s} at entry {i + 1}"
            )
    if dim < 0:
        dim = dim + len(inp0_shape) + 1
    dtypes = [t.dtype for t in tensors]
    dtype = dtypes[0]
    for dt in dtypes[1:]:
        dtype = torch.promote_types(dtype, dt)
    is_complex = dtype.is_complex
    if is_complex:
        complex_dtype = dtype
        dtype = torch.view_as_real(torch.empty((), dtype=dtype, device="cpu")).dtype
        tensors = [torch.view_as_real(t.to(complex_dtype)) for t in tensors]
    tensors = [t.to(dtype) if t.dtype != dtype else t for t in tensors]
    device = tensors[0].device
    work_shape = list(tensors[0].shape)
    out_shape = work_shape[:dim] + [len(tensors)] + work_shape[dim:]
    contig_strides = []
    stride = 1
    for dim_sz in reversed(out_shape):
        contig_strides.append(stride)
        stride *= dim_sz
    contig_strides.reverse()
    out = torch.empty_strided(out_shape, contig_strides, dtype=dtype, device=device)
    dim_prod_post = 1
    for s in work_shape[dim:]:
        dim_prod_post *= s
    if out.numel() == 0:
        return torch.view_as_complex(out) if is_complex else out
    BLOCK = 1024
    for tensor_idx, tensor in enumerate(tensors):
        tensor = tensor.contiguous()
        total_elements = tensor.numel()
        grid = (triton.cdiv(total_elements, BLOCK),)
        stack_copy_kernel[grid](
            out,
            tensor,
            len(tensors),
            dim_prod_post,
            tensor_idx,
            total_elements,
            BLOCK_X=BLOCK,
        )
    return torch.view_as_complex(out) if is_complex else out
