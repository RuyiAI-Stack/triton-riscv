import torch
import triton
import triton.language as tl

from .cat import cat


@triton.jit
def vstack_kernel(
    itensor_ptr0,
    itensor_ptr1,
    itensor_ptr2,
    itensor_ptr3,
    output_ptr,
    local_row0,
    local_row1,
    local_row2,
    local_row3,
    exc_row_offset0,
    exc_row_offset1,
    exc_row_offset2,
    exc_row_offset3,
    total_row_offset,
    row_stride,
    max_tile_elems,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    tensor_idx = tl.program_id(axis=1)
    col_idx = tl.arange(0, BLOCK_SIZE)

    intensor_ptr = tl.where(tensor_idx == 0, itensor_ptr0, itensor_ptr1)
    intensor_ptr = tl.where(tensor_idx == 2, itensor_ptr2, intensor_ptr)
    intensor_ptr = tl.where(tensor_idx == 3, itensor_ptr3, intensor_ptr)
    base_exc_row_idx = tl.where(
        tensor_idx == 0, exc_row_offset0, exc_row_offset1
    )
    base_exc_row_idx = tl.where(
        tensor_idx == 2, exc_row_offset2, base_exc_row_idx
    )
    base_exc_row_idx = tl.where(
        tensor_idx == 3, exc_row_offset3, base_exc_row_idx
    )
    local_row = tl.where(tensor_idx == 0, local_row0, local_row1)
    local_row = tl.where(tensor_idx == 2, local_row2, local_row)
    local_row = tl.where(tensor_idx == 3, local_row3, local_row)

    end_idx = local_row * row_stride.to(tl.int64)
    idx = (pid_x * BLOCK_SIZE + col_idx).to(tl.int64)
    offset_mask = idx < end_idx
    in_offset = intensor_ptr + idx
    row_stride_offset = (total_row_offset + base_exc_row_idx) * row_stride.to(
        tl.int64
    )
    out_offset = output_ptr + row_stride_offset + idx
    out = tl.load(in_offset, mask=offset_mask)
    tl.store(out_offset, out, mask=offset_mask)


def vstack(tensors: list):
    if len(tensors) == 0:
        raise RuntimeError("vstack expects a non-empty TensorList")
    return cat([torch.atleast_2d(tensor) for tensor in tensors], dim=0)
