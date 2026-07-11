import enum

import torch
import triton
import triton.language as tl


class MemOverlap(enum.Enum):
    No = 0
    Yes = 1
    TooHard = 2


def has_internal_overlapping(x: torch.Tensor):
    if x.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense(x):
        return MemOverlap.No
    for size, stride in zip(x.size(), x.stride()):
        if size > 1 and stride == 0:
            return MemOverlap.Yes
    return MemOverlap.TooHard


@triton.jit
def scatter_dim0_2d_kernel(
    inp,
    index,
    src,
    out,
    rows,
    cols,
    index_rows,
    BLOCK_COLS: tl.constexpr,
    REDUCE_ADD: tl.constexpr,
):
    output_row = tl.program_id(0)
    cols_offsets = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = cols_offsets < cols
    output_offsets = output_row * cols + cols_offsets
    value = tl.load(inp + output_offsets, mask=mask)

    for source_row in range(0, index_rows):
        source_offsets = source_row * cols + cols_offsets
        target_row = tl.load(index + source_offsets, mask=mask).to(tl.int32)
        source_value = tl.load(src + source_offsets, mask=mask)
        selected = target_row == output_row
        if REDUCE_ADD:
            value += tl.where(selected, source_value, 0.0)
        else:
            value = tl.where(selected, source_value, value)

    tl.store(out + output_offsets, value, mask=mask)


@triton.jit
def scatter_output_kernel(
    inp,
    index,
    src,
    out,
    input_shape,
    index_shape,
    index_strides,
    dim,
    scan_size,
    rank: tl.constexpr,
    REDUCE_ADD: tl.constexpr,
):
    output_index = tl.program_id(0)
    remaining = output_index
    index_base = tl.full((), 0, dtype=tl.int32)
    output_dim_coordinate = tl.full((), 0, dtype=tl.int32)
    other_dims_valid = tl.full((), 1, dtype=tl.int1)

    for axis in range(rank - 1, -1, -1):
        input_dim = tl.load(input_shape + axis)
        coordinate = remaining % input_dim
        remaining //= input_dim
        index_dim = tl.load(index_shape + axis)
        index_stride = tl.load(index_strides + axis)
        is_scatter_dim = axis == dim
        output_dim_coordinate = tl.where(
            is_scatter_dim, coordinate, output_dim_coordinate
        )
        index_base += tl.where(is_scatter_dim, 0, coordinate * index_stride)
        other_dims_valid &= is_scatter_dim | (coordinate < index_dim)

    value = tl.load(inp + output_index)
    scan_stride = tl.load(index_strides + dim)
    for scan_index in range(0, scan_size):
        source_offset = index_base + scan_index * scan_stride
        target = tl.load(index + source_offset).to(tl.int32)
        source_value = tl.load(src + source_offset)
        selected = other_dims_valid & (target == output_dim_coordinate)
        if REDUCE_ADD:
            value += tl.where(selected, source_value, 0.0)
        else:
            value = tl.where(selected, source_value, value)

    tl.store(out + output_index, value)


def _scatter_impl(inp, dim, index, src, reduce, inplace):
    dim = dim % inp.ndim
    if index.ndim != inp.ndim or src.ndim != inp.ndim:
        raise RuntimeError("Index tensor must have the same number of dimensions")
    reduce_add = reduce == "add"
    if reduce not in (None, "add"):
        raise RuntimeError(f"unsupported scatter reduction: {reduce}")

    input_contiguous = inp.contiguous()
    index_contiguous = index.contiguous()
    src_contiguous = src.contiguous()
    out = input_contiguous.clone()

    if inp.ndim == 2:
        transposed = dim == 1
        if transposed:
            input_contiguous = inp.transpose(0, 1).contiguous()
            index_contiguous = index.transpose(0, 1).contiguous()
            src_contiguous = src.transpose(0, 1).contiguous()
            out = input_contiguous.clone()
        rows, cols = input_contiguous.shape
        block_cols = 128
        scatter_dim0_2d_kernel[
            (rows, triton.cdiv(cols, block_cols))
        ](
            input_contiguous,
            index_contiguous,
            src_contiguous,
            out,
            rows,
            cols,
            index_contiguous.shape[0],
            BLOCK_COLS=block_cols,
            REDUCE_ADD=reduce_add,
        )
        if transposed:
            out = out.transpose(0, 1).contiguous()
    else:
        rank = inp.ndim
        input_shape = torch.tensor(
            inp.shape, dtype=torch.int32, device=inp.device
        )
        index_shape = torch.tensor(
            index.shape, dtype=torch.int32, device=inp.device
        )
        index_strides = torch.tensor(
            index_contiguous.stride(), dtype=torch.int32, device=inp.device
        )
        scatter_output_kernel[(out.numel(),)](
            input_contiguous,
            index_contiguous,
            src_contiguous,
            out,
            input_shape,
            index_shape,
            index_strides,
            dim,
            index.shape[dim],
            rank,
            REDUCE_ADD=reduce_add,
            num_warps=1,
        )

    if inplace:
        inp.copy_(out)
        return inp
    return out


def scatter(inp, dim, index, src, reduce=None):
    return _scatter_impl(inp, dim, index, src, reduce, False)


def scatter_(inp, dim, index, src, reduce=None):
    return _scatter_impl(inp, dim, index, src, reduce, True)
