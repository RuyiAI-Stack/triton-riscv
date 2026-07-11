import torch
import triton
import triton.language as tl


@triton.jit
def repeat_scalar_kernel(
    inp,
    out,
    input_shape,
    output_shape,
    input_strides,
    rank: tl.constexpr,
):
    output_index = tl.program_id(0)
    remaining = output_index
    input_index = tl.full((), 0, dtype=tl.int32)

    for dim in range(rank - 1, -1, -1):
        output_dim = tl.load(output_shape + dim)
        input_dim = tl.load(input_shape + dim)
        input_stride = tl.load(input_strides + dim)
        coordinate = remaining % output_dim
        remaining //= output_dim
        input_index += (coordinate % input_dim) * input_stride

    tl.store(out + output_index, tl.load(inp + input_index))


def repeat(inp: torch.Tensor, sizes) -> torch.Tensor:
    sizes = tuple(int(size) for size in sizes)
    if len(sizes) < inp.ndim:
        raise RuntimeError(
            "Number of dimensions of repeat dims can not be smaller than "
            "number of dimensions of tensor"
        )
    if any(size < 0 for size in sizes):
        raise RuntimeError("Trying to create tensor with negative dimension")

    rank = len(sizes)
    padded_shape = (1,) * (rank - inp.ndim) + tuple(inp.shape)
    output_shape = tuple(
        padded_shape[dim] * sizes[dim] for dim in range(rank)
    )
    inp = inp.contiguous()
    out = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
    if out.numel() == 0:
        return out

    strides = [1] * rank
    for dim in range(rank - 2, -1, -1):
        strides[dim] = strides[dim + 1] * padded_shape[dim + 1]

    input_shape_tensor = torch.tensor(
        padded_shape, dtype=torch.int32, device=inp.device
    )
    output_shape_tensor = torch.tensor(
        output_shape, dtype=torch.int32, device=inp.device
    )
    input_strides_tensor = torch.tensor(
        strides, dtype=torch.int32, device=inp.device
    )
    repeat_scalar_kernel[(out.numel(),)](
        inp,
        out,
        input_shape_tensor,
        output_shape_tensor,
        input_strides_tensor,
        rank,
        num_warps=1,
    )
    return out
