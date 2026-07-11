import torch
import triton
import triton.language as tl


def dim_compress(inp, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()


@triton.jit
def count_nonzero_kernel_1(
    x_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.int64)
    nonzero_count = tl.sum(is_nonzero, axis=0)
    tl.atomic_add(out_ptr, nonzero_count)


@triton.jit
def count_nonzero_kernel(
    x_ptr,
    out_ptr,
    N,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x = tl.program_id(0)

    nonzero_count = tl.full((), value=0, dtype=tl.int32)
    for start_n in range(0, N, BLOCK_SIZE):
        cols_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        offset = pid_x * N + cols_offsets
        mask = (offset < numel) & (cols_offsets < N)
        x = tl.load(x_ptr + offset, mask=mask, other=0)
        is_nonzero = (x != 0).to(tl.int32)
        nonzero_count += tl.sum(is_nonzero)

    tl.store(out_ptr + pid_x, nonzero_count.to(tl.int64))


@triton.jit
def count_nonzero_combin_kernel_1(
    x_ptr,
    out_ptr,
    N,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x = tl.program_id(0)
    nonzero_count = tl.full((), value=0, dtype=out_ptr.dtype.element_ty)
    for start_n in range(0, N, BLOCK_SIZE):
        cols_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        offset = pid_x * N + cols_offsets
        mask = offset < numel and cols_offsets < N
        x = tl.load(x_ptr + offset, mask=mask, other=0)
        nonzero_count += tl.sum(x)
    tl.store(out_ptr + pid_x, nonzero_count)


@triton.jit
def count_nonzero_combin_kernel(
    x_ptr,
    combin_ptr,
    N,
    combin_N,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    cols_offsets = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset = pid_x * N + cols_offsets
    mask = offset < numel and cols_offsets < N
    x = tl.load(x_ptr + offset, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.int64)
    nonzero_count = tl.sum(is_nonzero)
    tl.store(combin_ptr + pid_x * combin_N + pid_y, nonzero_count)


def count_nonzero(x, dim=None):
    if dim is not None:
        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        dim %= x.ndim
        shape = list(x.shape)
        x = dim_compress(x, dim)
        x = x.contiguous().flatten()
        numel = x.numel()
        reduction_size = shape[dim]
        out_shape = list(shape)
        del out_shape[dim]
        out = torch.zeros(out_shape, dtype=torch.int64, device=x.device)
        grid = (numel // reduction_size,)
        count_nonzero_kernel[grid](
            x, out, reduction_size, numel, BLOCK_SIZE=1024
        )
        return out

    x = x.contiguous().flatten()
    numel = x.numel()
    out = torch.zeros(1, dtype=torch.int64, device=x.device)
    count_nonzero_kernel[(1,)](
        x, out, numel, numel, BLOCK_SIZE=1024
    )
    return out[0]
