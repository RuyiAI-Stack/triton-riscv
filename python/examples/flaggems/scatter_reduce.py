import torch
import triton
import triton.language as tl


@triton.jit
def scatter_reduce_output_kernel(
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
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
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

    input_value = tl.load(inp + output_index)
    if REDUCE == 0 or REDUCE == 2:
        value = input_value if INCLUDE_SELF else 0.0
    elif REDUCE == 1:
        value = input_value if INCLUDE_SELF else 1.0
    elif REDUCE == 3:
        value = input_value if INCLUDE_SELF else float("-inf")
    else:
        value = input_value if INCLUDE_SELF else float("inf")
    contribution_count = tl.full(
        (), 1 if INCLUDE_SELF else 0, dtype=tl.int32
    )

    scan_stride = tl.load(index_strides + dim)
    for scan_index in range(0, scan_size):
        source_offset = index_base + scan_index * scan_stride
        target = tl.load(index + source_offset).to(tl.int32)
        source_value = tl.load(src + source_offset)
        selected = other_dims_valid & (target == output_dim_coordinate)
        contribution_count += selected.to(tl.int32)
        if REDUCE == 0 or REDUCE == 2:
            value += tl.where(selected, source_value, 0.0)
        elif REDUCE == 1:
            value *= tl.where(selected, source_value, 1.0)
        elif REDUCE == 3:
            value = tl.where(selected & (source_value > value), source_value, value)
        else:
            value = tl.where(selected & (source_value < value), source_value, value)

    has_contribution = contribution_count > 0
    if REDUCE == 2:
        value = tl.where(
            has_contribution, value / contribution_count, input_value
        )
    elif not INCLUDE_SELF:
        value = tl.where(has_contribution, value, input_value)
    tl.store(out + output_index, value)


_REDUCTIONS = {"sum": 0, "prod": 1, "mean": 2, "amax": 3, "amin": 4}


def scatter_reduce(inp, dim, index, src, reduce, *, include_self=True):
    if reduce not in _REDUCTIONS:
        raise RuntimeError(f"unsupported scatter reduction: {reduce}")
    dim = dim % inp.ndim
    inp_contiguous = inp.contiguous()
    index_contiguous = index.contiguous()
    src_contiguous = src.contiguous()
    out = torch.empty_like(inp_contiguous)
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
    scatter_reduce_output_kernel[(out.numel(),)](
        inp_contiguous,
        index_contiguous,
        src_contiguous,
        out,
        input_shape,
        index_shape,
        index_strides,
        dim,
        index.shape[dim],
        rank,
        REDUCE=_REDUCTIONS[reduce],
        INCLUDE_SELF=include_self,
        num_warps=1,
    )
    return out


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    result = scatter_reduce(
        inp, dim, index, src, reduce, include_self=include_self
    )
    inp.copy_(result)
    return inp


def scatter_reduce_out(
    inp, dim, index, src, reduce, *, include_self=True, out=None
):
    result = scatter_reduce(
        inp, dim, index, src, reduce, include_self=include_self
    )
    if out is not None:
        out.copy_(result)
        return out
    return result
