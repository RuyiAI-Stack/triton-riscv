import torch
import triton
import triton.language as tl


@triton.jit
def hstack_copy_func_kernel(
    out_ptr,
    in_ptr,
    dim_size_in,
    dim_size_out,
    dim_prod_post,
    dim_offset,
    total_elements,
    BLOCK_X: tl.constexpr,
):
    pid_x = tl.program_id(0)

    block_start = pid_x * BLOCK_X
    offsets = tl.arange(0, BLOCK_X)
    mask = block_start + offsets < total_elements

    idx = block_start + offsets

    pre_idx = idx // (dim_size_in * dim_prod_post)
    dim_idx = (idx // dim_prod_post) % dim_size_in
    post_idx = idx % dim_prod_post

    out_idx = (
        pre_idx * dim_size_out * dim_prod_post
        + (dim_idx + dim_offset) * dim_prod_post
        + post_idx
    )

    data = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + out_idx, data, mask=mask)


def hstack(
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> torch.Tensor:
    if len(tensors) == 0:
        raise RuntimeError("hstack expected a non-empty TensorList")

    if tensors[0].ndim == 0:
        tensors[0] = tensors[0].view(1)
    inp0_shape = tensors[0].shape
    out_shape = list(inp0_shape)
    inp_shapes = [inp0_shape]

    if len(inp0_shape) == 1:
        dim = 0
    else:
        dim = 1

    for tensor_num, tensor in enumerate(tensors[1:]):
        if tensor.ndim == 0:
            tensor = tensor.view(1)
        if tensor.ndim != tensors[0].ndim:
            raise RuntimeError(
                f"Tensors must have same number of dimensions: "
                f"got {tensors[0].ndim} and {tensor.ndim}"
            )

        inp_shape = tensor.shape
        inp_shapes.append(inp_shape)

        for i in range(len(inp_shape)):
            if i != dim and inp_shape[i] != inp0_shape[i]:
                raise RuntimeError(
                    "Sizes of tensors must match except in dimension "
                    f"{dim}. Expected size {inp0_shape[i]} but got "
                    f"size {inp_shape[i]} for tensor number "
                    f"{tensor_num + 1} in the list."
                )

    inp_shapes = [list(_.shape) for _ in tensors]
    inp0_shape = inp_shapes[0]

    # Type promotion: find the common dtype for all tensors
    dtypes = [t.dtype for t in tensors]
    dtype = dtypes[0]
    for dt in dtypes[1:]:
        dtype = torch.promote_types(dtype, dt)
    # Convert all tensors to the common dtype if needed
    tensors = [t.to(dtype) if t.dtype != dtype else t for t in tensors]
    device = tensors[0].device
    out_shape[dim] = sum(s[dim] for s in inp_shapes)
    out = torch.empty(out_shape, dtype=dtype, device=device)

    dim_prod_post = 1
    for s in inp0_shape[dim + 1 :]:
        dim_prod_post *= s
    BLOCK = 1024
    dim_offset = 0
    for tensor in tensors:
        tensor = tensor.contiguous()
        shape = tensor.shape
        total_elements = tensor.numel()
        dim_size_in = shape[dim]
        dim_size_out = out_shape[dim]
        grid = (triton.cdiv(total_elements, BLOCK),)

        hstack_copy_func_kernel[grid](
            out,
            tensor,
            dim_size_in,
            dim_size_out,
            dim_prod_post,
            dim_offset,
            total_elements,
            BLOCK_X=BLOCK,
        )

        dim_offset += dim_size_in

    return out
