import torch
import triton
import triton.language as tl


@triton.jit
def stack_copy_func_kernel(
    out_ptr,
    in_ptr,
    dim_size_out,
    dim_prod_post,
    dim_offset,
    total_elements,
    BLOCK_X: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid.to(tl.int64) * BLOCK_X
    offsets = tl.arange(0, BLOCK_X).to(tl.int64)
    idx = block_start + offsets
    mask = idx < total_elements

    pre_idx = idx // dim_prod_post
    post_idx = idx % dim_prod_post

    out_idx = (
        pre_idx * dim_size_out * dim_prod_post + dim_offset * dim_prod_post + post_idx
    )

    data = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + out_idx, data, mask=mask)


def stack(
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor], dim: int = 0
) -> torch.Tensor:
    if len(tensors) == 0:
        raise RuntimeError("stack expected a non-empty TensorList")

    inp_shapes = [list(_.shape) for _ in tensors]
    inp0_shape = inp_shapes[0]
    for i, s in enumerate(inp_shapes[1:]):
        ndim = tensors[i + 1].dim()
        if (dim < -ndim - 1) or (dim > ndim):
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim - 1}, {ndim}], but got {dim})"
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
    tensors = [t.to(dtype) if t.dtype != dtype else t for t in tensors]
    device = tensors[0].device
    out_shape = [*inp0_shape[:dim], len(tensors), *inp0_shape[dim:]]
    out = torch.empty(out_shape, dtype=dtype, device=device)

    dim_prod_post = 1
    for s in inp0_shape[dim:]:
        dim_prod_post *= s

    BLOCK = 1024
    dim_size_out = len(tensors)
    for i, tensor in enumerate(tensors):
        tensor = tensor.contiguous()
        total_elements = tensor.numel()
        if total_elements == 0:
            continue
        grid = (triton.cdiv(total_elements, BLOCK),)
        stack_copy_func_kernel[grid](
            out,
            tensor,
            dim_size_out,
            dim_prod_post,
            i,
            total_elements,
            BLOCK_X=BLOCK,
        )

    return out
