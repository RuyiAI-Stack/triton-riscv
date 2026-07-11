import torch
import triton
import triton.language as tl


@triton.jit
def cat_copy_kernel(
    out_ptr,
    in_ptr,
    inner_size,
    out_inner_size,
    out_inner_offset,
    BLOCK_SIZE: tl.constexpr,
):
    pre_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < inner_size

    input_offsets = pre_idx * inner_size + offsets
    output_offsets = pre_idx * out_inner_size + out_inner_offset + offsets
    data = tl.load(in_ptr + input_offsets, mask=mask)
    tl.store(out_ptr + output_offsets, data, mask=mask)


def _cat_run_kernel(
    A: list[torch.Tensor],
    dim: int,
    out_shape: list[int],
    out: torch.Tensor,
):
    BLOCK_SIZE = 1024
    dim_size_out = out_shape[dim]
    dim_prod_post = 1
    for d in range(dim + 1, A[0].ndim):
        dim_prod_post *= A[0].shape[d]

    pre_count = 1
    for d in range(dim):
        pre_count *= A[0].shape[d]

    dim_offset = 0
    for tensor in A:
        tensor = tensor.contiguous()
        dim_size_in = tensor.shape[dim]
        inner_size = dim_size_in * dim_prod_post
        out_inner_size = dim_size_out * dim_prod_post
        out_inner_offset = dim_offset * dim_prod_post
        grid = (pre_count, triton.cdiv(inner_size, BLOCK_SIZE))
        cat_copy_kernel[grid](
            out,
            tensor,
            inner_size,
            out_inner_size,
            out_inner_offset,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        dim_offset += dim_size_in


def _cat_build_working_list(
    A: tuple[torch.Tensor, ...] | list[torch.Tensor], dim: int
):
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
                f"Tensors must have same number of dimensions: got {len(inp0_shape)} and {len(s)}"
            )
    for tensor_idx, inp_shape in enumerate(inp_shapes):
        for idx, (common_length, length) in enumerate(
            zip(inp0_shape, inp_shape)
        ):
            if idx != dim and length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
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
        raise RuntimeError(
            f"cat.out: expected out dtype {dtype}, got {out.dtype}"
        )
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
