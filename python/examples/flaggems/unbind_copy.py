from typing import List

import torch
import triton
import triton.language as tl

# unbind_copy: aten::unbind as value copies. One 2D-grid Triton launch
# writes every slice into a single flat storage (axis 1 = slice id), then
# each slice is viewed out of it. The upstream autotuner config is replaced
# by the fixed 1024 block used by the sibling copy kernels.


@triton.jit
def unbind_copy_kernel(
    input_ptr,
    output_ptr,
    dim_size,
    dim_prod_post,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x = tl.program_id(0)
    slice_id = tl.program_id(1)
    block_start = pid_x * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < num_elements
    idx = block_start + offsets
    pre_idx = idx // dim_prod_post
    post_idx = idx % dim_prod_post
    input_flat_idx = (
        pre_idx * dim_size * dim_prod_post + slice_id * dim_prod_post + post_idx
    )
    data = tl.load(input_ptr + input_flat_idx, mask=mask)
    slice_offset = slice_id * num_elements
    tl.store(output_ptr + slice_offset + idx, data, mask=mask)


def unbind_copy(input: torch.Tensor, dim: int = 0) -> List[torch.Tensor]:
    """Unbind into a list of fresh slice copies along ``dim`` (rank drops by one)."""
    if dim < 0:
        dim = dim + input.ndim
    if dim < 0 or dim >= input.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-input.ndim}, {input.ndim - 1}], but got {dim})"
        )
    num_slices = input.shape[dim]
    if num_slices == 0:
        return []
    output_shape = list(input.shape)
    del output_shape[dim]

    dim_prod_post = 1
    for d in range(dim + 1, input.ndim):
        dim_prod_post *= input.shape[d]
    num_elements_per_slice = 1
    for s in output_shape:
        num_elements_per_slice *= s

    input = input.contiguous()
    output_storage = torch.empty(
        num_slices * num_elements_per_slice, dtype=input.dtype, device=input.device
    )

    BLOCK = 1024
    grid_x = triton.cdiv(num_elements_per_slice, BLOCK)
    grid = (grid_x, num_slices)

    unbind_copy_kernel[grid](
        input,
        output_storage,
        num_slices,
        dim_prod_post,
        num_elements_per_slice,
        BLOCK_SIZE=BLOCK,
    )

    outputs = []
    for i in range(num_slices):
        start = i * num_elements_per_slice
        end = start + num_elements_per_slice
        outputs.append(output_storage[start:end].reshape(output_shape))

    return outputs
