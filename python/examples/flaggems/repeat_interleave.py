import torch
import triton
import triton.language as tl


def repeat_interleave_self_int(inp, repeats, dim=None, *, output_size=None):
    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if (dim < -inp.ndim) or (dim >= inp.ndim):
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
            )
    inp_shape = list(inp.shape)
    inp_stride = list(inp.stride())
    output_shape = list(inp.shape)

    if dim < 0:
        dim = dim + len(inp_shape)

    output_shape[dim] *= repeats

    if output_size is not None and output_size != output_shape[dim]:
        raise RuntimeError(
            f"repeat_interleave: Invalid output_size, expected {output_shape[dim]} but got {output_size}"
        )

    output = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)

    if repeats == 0:
        return output

    in_view_stride = [*inp_stride[: dim + 1], 0, *inp_stride[dim + 1 :]]
    out_view_shape = [*inp_shape[: dim + 1], repeats, *inp_shape[dim + 1 :]]

    in_view = torch.as_strided(inp, out_view_shape, in_view_stride)
    in_view = in_view.contiguous()
    out_view = output.reshape(out_view_shape)
    out_view.copy_(in_view)
    return output


@triton.jit
def repeat_interleave_tensor_kernel(
    repeats_ptr,
    cumsum_ptr,
    out_ptr,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    mask = pid < size
    cumsum = tl.load(cumsum_ptr + pid, mask=mask, other=0)
    repeats = tl.load(repeats_ptr + pid, mask=mask, other=0)
    out_offset = cumsum - repeats

    out_ptr += out_offset
    for start_k in range(0, repeats, BLOCK_SIZE):
        offsets_k = start_k + tl.arange(0, BLOCK_SIZE)
        mask_k = offsets_k < repeats
        tl.store(out_ptr + offsets_k, pid, mask=mask_k)


def repeat_interleave_tensor(repeats, *, output_size=None):
    assert repeats.ndim == 1, (
        "repeat_interleave only accept 1D vector as repeat"
    )

    cumsum = repeats.cumsum(axis=0)
    result_size = cumsum[-1].item() if repeats.numel() > 0 else 0

    assert result_size >= 0, "repeats can not be negative"

    if result_size == 0:
        return torch.empty((0,), dtype=repeats.dtype, device=repeats.device)

    out = torch.empty(
        (result_size,), dtype=repeats.dtype, device=repeats.device
    )
    size = repeats.size(0)

    grid = (size,)
    BLOCK_SIZE = 32
    repeat_interleave_tensor_kernel[grid](
        repeats,
        cumsum,
        out,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )
    return out


def repeat_interleave_self_tensor(inp, repeats, dim=None, *, output_size=None):
    if repeats.numel() == 0:
        return inp.clone()

    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if (dim < -inp.ndim) or (dim >= inp.ndim):
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
            )

    if repeats.ndim == 0 or (repeats.ndim == 1 and repeats.size(0) == 1):
        return repeat_interleave_self_int(
            inp, repeats.item(), dim=dim, output_size=output_size
        )
    elif repeats.ndim > 1:
        raise RuntimeError("repeats must be 0-dim or 1-dim tensor")

    inp_shape = list(inp.shape)
    if dim < 0:
        dim = dim + len(inp_shape)

    if repeats.size(0) != inp_shape[dim]:
        raise RuntimeError(
            f"repeats must have the same size as input along dim, but got \
                repeats.size(0) = {repeats.size(0)} and input.size({dim}) = {inp_shape[dim]}"
        )

    indices = repeat_interleave_tensor(repeats)
    res = torch.index_select(inp, dim, indices)
    return res


def repeat_interleave(input, repeats, dim=None, *, output_size=None):
    if isinstance(repeats, int):
        return repeat_interleave_self_int(
            input, repeats, dim=dim, output_size=output_size
        )
    else:
        return repeat_interleave_self_tensor(
            input, repeats, dim=dim, output_size=output_size
        )
