from operator import index

import torch
import triton
import triton.language as tl


@triton.jit
def split_copy_kernel(
    input_ptr,
    output_ptr,
    num_elements: tl.constexpr,
    dim_size_input: tl.constexpr,
    dim_size_output: tl.constexpr,
    split_start,
    dim_prod_post: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel to copy a split from input to output tensor."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offset + offsets < num_elements

    idx = offset + offsets
    pre_idx = idx // (dim_size_output * dim_prod_post)
    split_idx = (idx // dim_prod_post) % dim_size_output
    post_idx = idx % dim_prod_post

    input_idx = (
        pre_idx * dim_size_input * dim_prod_post
        + (split_start + split_idx) * dim_prod_post
        + post_idx
    )

    data = tl.load(input_ptr + input_idx, mask=mask)
    tl.store(output_ptr + idx, data, mask=mask)


def _invalid_tensor_split_arguments(argument_type: str):
    raise TypeError(
        "tensor_split() received an invalid combination of arguments - got "
        f"(Tensor, {argument_type}), but expected one of:\n"
        " * (Tensor input, tuple of ints indices, int dim = 0)\n"
        " * (Tensor input, Tensor tensor_indices_or_sections, int dim = 0)\n"
        " * (Tensor input, int sections, int dim = 0)\n"
    )


def tensor_split(
    input: torch.Tensor,
    indices_or_sections: int | list[int] | torch.Tensor,
    dim: int = 0,
) -> list[torch.Tensor]:
    """Split a tensor into copies along a given dimension using a Triton kernel."""
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Expected tensor, got {type(input)}")

    ndim = input.ndim
    if ndim == 0:
        raise RuntimeError(
            "tensor_split expected at least a 1-dimensional tensor, but got a "
            "tensor with 0 dimensions"
        )
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], got {dim})"
        )
    dim %= ndim
    dim_size = input.shape[dim]

    if isinstance(indices_or_sections, torch.Tensor):
        if indices_or_sections.dtype != torch.long:
            raise RuntimeError(
                "tensor_split expected tensor_indices_or_sections to have dtype of long"
            )
        if indices_or_sections.ndim > 1:
            raise RuntimeError(
                "tensor_split expected tensor_indices_or_sections to be "
                "zero-dimensional or one-dimensional"
            )
        indices_or_sections = (
            indices_or_sections.item()
            if indices_or_sections.ndim == 0
            else indices_or_sections.tolist()
        )

    if isinstance(indices_or_sections, bool):
        _invalid_tensor_split_arguments("bool")

    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
        if sections <= 0:
            raise RuntimeError(
                f"number of sections must be larger than 0, got {sections}"
            )

        base_size, remainder = divmod(dim_size, sections)
        split_sizes = [
            base_size + (split_idx < remainder) for split_idx in range(sections)
        ]
        split_starts = []
        start = 0
        for split_size in split_sizes:
            split_starts.append(start)
            start += split_size
    else:
        argument_type = type(indices_or_sections).__name__
        if isinstance(indices_or_sections, tuple):
            indices_or_sections = list(indices_or_sections)
        if not isinstance(indices_or_sections, list):
            _invalid_tensor_split_arguments(argument_type)

        indices = []
        for split_index in indices_or_sections:
            if isinstance(split_index, bool):
                _invalid_tensor_split_arguments(argument_type)
            try:
                indices.append(index(split_index))
            except TypeError:
                _invalid_tensor_split_arguments(argument_type)

        normalized_indices = []
        for split_index in indices:
            if split_index < 0:
                split_index += dim_size
            normalized_indices.append(min(max(split_index, 0), dim_size))

        split_sizes = []
        split_starts = []
        previous_index = 0
        for split_index in normalized_indices:
            split_starts.append(previous_index)
            split_sizes.append(max(split_index - previous_index, 0))
            previous_index = split_index
        split_starts.append(previous_index)
        split_sizes.append(max(dim_size - previous_index, 0))

    output_tensors = []
    dim_prod_post = 1
    for trailing_dim in range(dim + 1, ndim):
        dim_prod_post *= input.shape[trailing_dim]

    block_size = 1024
    input_contiguous = input.contiguous()
    for split_start, split_size in zip(split_starts, split_sizes):
        output_shape = list(input.shape)
        output_shape[dim] = split_size
        output_tensor = torch.empty(
            output_shape, dtype=input.dtype, device=input.device
        )

        if split_size != 0:
            total_elements = output_tensor.numel()
            grid = (triton.cdiv(total_elements, block_size),)
            split_copy_kernel[grid](
                input_contiguous,
                output_tensor,
                total_elements,
                dim_size,
                split_size,
                split_start,
                dim_prod_post,
                BLOCK_SIZE=block_size,
            )

        output_tensors.append(output_tensor)

    return output_tensors
