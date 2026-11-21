from typing import List

import torch
import triton
import triton.language as tl

# _chunk_cat: aten::_chunk_cat -- chunk every tensor along ``dim`` into
# ``num_chunks`` equal parts (zero-padded) and concatenate across tensors
# per chunk index. The single-tensor path runs three Triton gather kernels
# (1d / 2d-dim0 / 2d-dim1) plus an N-d stride-decomposed kernel; the
# multi-tensor path is host chunk/pad/stack/cat composition (upstream
# behavior, kept verbatim).


@triton.jit
def _chunk_cat_kernel_1d(
    input_ptr,
    output_ptr,
    input_size,
    chunk_size,
    total_output_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < total_output_elements
    out_idx = block_start + offsets
    chunk_idx = out_idx // chunk_size
    within_chunk = out_idx % chunk_size
    orig_idx = chunk_idx * chunk_size + within_chunk
    data = tl.load(input_ptr + orig_idx, mask=mask & (orig_idx < input_size))
    tl.store(output_ptr + out_idx, data, mask=mask)


@triton.jit
def _chunk_cat_kernel_2d_dim0(
    input_ptr,
    output_ptr,
    input_size_0,
    input_size_1,
    chunk_size,
    total_output_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < total_output_elements
    out_idx = block_start + offsets
    remaining = input_size_1
    chunk_idx = out_idx // (chunk_size * remaining)
    remainder = out_idx % (chunk_size * remaining)
    row_offset = remainder // remaining
    col = remainder % remaining
    orig_row = chunk_idx * chunk_size + row_offset
    orig_idx = orig_row * remaining + col
    valid = mask & (orig_row < input_size_0)
    data = tl.load(input_ptr + orig_idx, mask=valid)
    tl.store(output_ptr + out_idx, data, mask=mask)


@triton.jit
def _chunk_cat_kernel_2d_dim1(
    input_ptr,
    output_ptr,
    input_size_0,
    input_size_1,
    num_chunks,
    chunk_size,
    total_output_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < total_output_elements
    out_idx = block_start + offsets
    row = out_idx // (num_chunks * chunk_size)
    remainder = out_idx % (num_chunks * chunk_size)
    chunk_idx = remainder // chunk_size
    within_chunk = remainder % chunk_size
    orig_col = chunk_idx * chunk_size + within_chunk
    orig_idx = row * input_size_1 + orig_col
    valid = mask & (orig_col < input_size_1)
    data = tl.load(input_ptr + orig_idx, mask=valid)
    tl.store(output_ptr + out_idx, data, mask=mask)


@triton.jit
def _chunk_cat_kernel_nd(
    input_ptr,
    output_ptr,
    dim_size,
    num_chunks,
    chunk_size,
    stride_before,
    stride_after,
    total_output_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    out_idx = block_start + offsets
    mask = out_idx < total_output_elements
    # Output layout: [...before..., num_chunks, chunk_size, ...after...]
    chunk_block = chunk_size * stride_after
    full_block = num_chunks * chunk_block
    before_idx = out_idx // full_block
    remainder = out_idx % full_block
    chunk_idx = remainder // chunk_block
    remainder2 = remainder % chunk_block
    within = remainder2 // stride_after
    after_idx = remainder2 % stride_after
    orig_dim_idx = chunk_idx * chunk_size + within
    orig_idx = (
        before_idx * (dim_size * stride_after) + orig_dim_idx * stride_after + after_idx
    )
    valid = mask & (orig_dim_idx < dim_size)
    data = tl.load(input_ptr + orig_idx, mask=valid, other=0.0)
    tl.store(output_ptr + out_idx, data, mask=mask)


def _chunk_cat_triton_1d(tensor, dim, num_chunks):
    input_size = tensor.shape[0]
    chunk_size = (input_size + num_chunks - 1) // num_chunks
    output_shape = [num_chunks, chunk_size]
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    BLOCK = 1024
    total_output = num_chunks * chunk_size
    grid = (triton.cdiv(total_output, BLOCK),)
    _chunk_cat_kernel_1d[grid](
        tensor, output, input_size, chunk_size, total_output, BLOCK_SIZE=BLOCK
    )
    return output


def _chunk_cat_triton_2d(tensor, dim, num_chunks):
    input_size_0, input_size_1 = tensor.shape
    chunk_size = (
        (input_size_0 + num_chunks - 1) // num_chunks
        if dim == 0
        else (input_size_1 + num_chunks - 1) // num_chunks
    )
    if dim == 0:
        remaining = input_size_1
        output_shape = [num_chunks, chunk_size * remaining]
    else:
        output_shape = [input_size_0, num_chunks, chunk_size]
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    BLOCK = 1024
    total_output = 1
    for s in output_shape:
        total_output *= s
    grid = (triton.cdiv(total_output, BLOCK),)
    if dim == 0:
        _chunk_cat_kernel_2d_dim0[grid](
            tensor,
            output,
            input_size_0,
            input_size_1,
            chunk_size,
            total_output,
            BLOCK_SIZE=BLOCK,
        )
    else:
        _chunk_cat_kernel_2d_dim1[grid](
            tensor,
            output,
            input_size_0,
            input_size_1,
            num_chunks,
            chunk_size,
            total_output,
            BLOCK_SIZE=BLOCK,
        )
    return output


def _chunk_cat_triton_nd(tensor, dim, num_chunks):
    dim_size = tensor.shape[dim]
    chunk_size = (dim_size + num_chunks - 1) // num_chunks
    stride_before = 1
    for i in range(dim):
        stride_before *= tensor.shape[i]
    stride_after = 1
    for i in range(dim + 1, tensor.ndim):
        stride_after *= tensor.shape[i]
    # aten merges the chunk and trailing dims: before + [num_chunks] +
    # [chunk_size * after]; the kernel's row-major flat layout is identical.
    output_shape = (
        list(tensor.shape[:dim])
        + [num_chunks, chunk_size * stride_after]
        + list(tensor.shape[dim + 1 :])[0:0]
    )
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    BLOCK = 1024
    total_output = output.numel()
    grid = (triton.cdiv(total_output, BLOCK),)
    tensor = tensor.contiguous()
    _chunk_cat_kernel_nd[grid](
        tensor,
        output,
        dim_size,
        num_chunks,
        chunk_size,
        stride_before,
        stride_after,
        total_output,
        BLOCK_SIZE=BLOCK,
    )
    return output


def _chunk_cat_triton(tensor: torch.Tensor, dim: int, num_chunks: int) -> torch.Tensor:
    dim = dim % tensor.ndim
    if tensor.numel() == 0:
        dim_size = tensor.shape[dim]
        chunk_size = (dim_size + num_chunks - 1) // num_chunks if dim_size > 0 else 0
        output_shape = (
            list(tensor.shape[:dim])
            + [num_chunks, chunk_size]
            + list(tensor.shape[dim + 1 :])
        )
        return torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    if tensor.ndim == 1:
        return _chunk_cat_triton_1d(tensor, dim, num_chunks)
    elif tensor.ndim == 2:
        return _chunk_cat_triton_2d(tensor, dim, num_chunks)
    else:
        return _chunk_cat_triton_nd(tensor, dim, num_chunks)


def _chunk_cat_impl(
    tensors: List[torch.Tensor], dim: int, num_chunks: int
) -> torch.Tensor:
    if len(tensors) == 0:
        raise ValueError("_chunk_cat(): expected a non-empty list of Tensors")
    if num_chunks <= 0:
        raise ValueError(f"_chunk_cat(): num_chunks must be positive, got {num_chunks}")
    ndim = tensors[0].ndim
    if ndim == 0:
        raise ValueError("_chunk_cat(): expected tensors with at least 1 dimension")
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"_chunk_cat(): dim {dim} out of range for tensor with {ndim} dimensions"
        )
    if len(tensors) == 1:
        return _chunk_cat_triton(tensors[0], dim, num_chunks)
    # aten semantics (probed): the multi-tensor result is the concatenation
    # of each tensor's single-tensor chunk-stack along the axis right after
    # the chunk axis. The upstream host composition (chunk/pad/stack/slice)
    # does not reproduce this layout and indexes out of bounds for dim > 0.
    dim = dim % ndim
    results = [_chunk_cat_triton(t, dim, num_chunks) for t in tensors]
    return torch.cat(results, dim=dim + 1)


def chunk_cat(tensors: List[torch.Tensor], dim: int, num_chunks: int) -> torch.Tensor:
    """Chunk each tensor into ``num_chunks`` parts and concat across tensors per chunk."""
    return _chunk_cat_impl(tensors, dim, num_chunks)
