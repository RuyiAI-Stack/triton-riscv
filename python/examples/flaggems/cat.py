import torch
import triton
import triton.language as tl


@triton.jit
def cat_copy_func_kernel(
    out_ptr,
    in_ptr,
    dim_size_in,
    dim_size_out,
    dim_prod_post,
    dim_offset: tl.int64,
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


def _cat_run_kernel(
    A: list[torch.Tensor],
    dim: int,
    out_shape: list[int],
    out: torch.Tensor,
):
    BLOCK = 1024
    dim_offset = 0
    for tensor in A:
        tensor = tensor.contiguous()
        dim_size_in = tensor.shape[dim]
        total_elements = tensor.numel()
        if total_elements == 0:
            dim_offset += dim_size_in
            continue

        dim_prod_post = 1
        for d in range(dim + 1, A[0].ndim):
            dim_prod_post *= A[0].shape[d]

        grid = (triton.cdiv(total_elements, BLOCK),)
        cat_copy_func_kernel[grid](
            out,
            tensor,
            dim_size_in,
            out_shape[dim],
            dim_prod_post,
            dim_offset,
            total_elements,
            BLOCK_X=BLOCK,
        )
        dim_offset += dim_size_in


def _cat_build_working_list(A: tuple[torch.Tensor, ...] | list[torch.Tensor], dim: int):
    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return "single", A[0]

    device = A[0].device
    dtype = A[0].dtype
    A = list(A)
    for i in range(len(A) - 1, -1, -1):
        if A[i].shape == torch.Size([0]):
            A.pop(i)
    if len(A) == 0:
        return "empty", torch.tensor([], device=device, dtype=dtype)
    if len(A) == 1:
        return "single", A[0]

    assert dim >= -A[0].ndim and dim < A[0].ndim, f"Invalid dim: {dim}"
    dim %= A[0].ndim

    inp_shapes = [list(_.shape) for _ in A]
    inp0_shape = inp_shapes[0]
    for s in inp_shapes[1:]:
        if len(s) != len(inp0_shape):
            raise RuntimeError(
                "Tensors must have same number of dimensions: "
                f"got {len(inp0_shape)} and {len(s)}"
            )
    for tensor_idx, inp_shape in enumerate(inp_shapes):
        for idx, (common_length, length) in enumerate(zip(inp0_shape, inp_shape)):
            if idx != dim and length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} "
                    "for tensor number "
                    f"{tensor_idx} in the list"
                )

    dtypes = [t.dtype for t in A]
    dtype = dtypes[0]
    for dt in dtypes[1:]:
        dtype = torch.promote_types(dtype, dt)
    A = [t.to(dtype) if t.dtype != dtype else t for t in A]

    shapes = [t.shape for t in A]
    cat_dim_sizes = [s[dim] for s in shapes]
    out_shape = list(shapes[0])
    out_shape[dim] = sum(cat_dim_sizes)
    return "multi", (A, dim, out_shape, dtype, A[0].device)


def cat_out(
    A: tuple[torch.Tensor, ...] | list[torch.Tensor],
    dim: int = 0,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    mode, payload = _cat_build_working_list(A, dim)
    if mode == "single":
        t = payload
        out.resize_(t.shape)
        if out.dtype != t.dtype:
            out.copy_(t.to(out.dtype))
        else:
            out.copy_(t)
        return out
    if mode == "empty":
        t = payload
        out.resize_(t.shape)
        out.copy_(t)
        return out

    A, dim, out_shape, dtype, _ = payload
    if out.dtype != dtype:
        raise RuntimeError(f"cat.out: expected out dtype {dtype}, got {out.dtype}")
    if list(out.shape) != out_shape:
        out.resize_(out_shape)
    _cat_run_kernel(A, dim, out_shape, out)
    return out


def cat(A, dim=0):
    mode, payload = _cat_build_working_list(A, dim)
    if mode == "single":
        return payload
    if mode == "empty":
        return payload

    A, dim, out_shape, dtype, device = payload
    out = torch.empty(out_shape, dtype=dtype, device=device)
    _cat_run_kernel(A, dim, out_shape, out)
    return out
