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
    mid_ptr,
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
    tl.store(mid_ptr + pid, nonzero_count)


@triton.jit
def count_nonzero_kernel_2(
    mid_ptr,
    out_ptr,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    counts = tl.load(mid_ptr + offsets, mask=mask, other=0)
    total = tl.sum(counts, axis=0)
    tl.store(out_ptr, total)


@triton.jit
def count_nonzero_kernel(
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
        is_nonzero = (x != 0).to(tl.int64)
        nonzero_count += tl.sum(is_nonzero)

    tl.store(out_ptr + pid_x, nonzero_count)


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
        shape = x.shape
        BLOCK_SIZE = 2048
        numel = x.numel()
        x = dim_compress(x, dim)
        x = x.contiguous().flatten()
        combin_shape = list(shape)
        combin_shape[dim] = triton.cdiv(combin_shape[dim], BLOCK_SIZE)
        if combin_shape[dim] != 1:
            combin = torch.zeros(combin_shape, dtype=torch.int64, device=x.device)
            grid = (triton.cdiv(numel, shape[dim]), combin_shape[dim], 1)
            count_nonzero_combin_kernel[grid](
                x, combin, shape[dim], combin_shape[dim], numel, BLOCK_SIZE
            )
            x = combin
            shape = x.shape
            numel = x.numel()
            out_shape = list(shape)
            del out_shape[dim]
            out = torch.zeros(out_shape, dtype=torch.int64, device=x.device)

            grid = (triton.cdiv(numel, shape[dim]),)

            count_nonzero_combin_kernel_1[grid](
                x, out, shape[dim], numel, BLOCK_SIZE=2048
            )
            return out
        out_shape = list(shape)
        del out_shape[dim]
        out = torch.zeros(out_shape, dtype=torch.int64, device=x.device)

        grid2 = (triton.cdiv(numel, shape[dim]),)

        count_nonzero_kernel[grid2](x, out, shape[dim], numel, BLOCK_SIZE=2048)
        return out
    else:
        x = x.contiguous().flatten()
        numel = x.numel()

        out = torch.zeros(1, dtype=torch.int64, device=x.device)

        BLOCK_SIZE = 1024

        num_blocks = triton.cdiv(numel, BLOCK_SIZE)
        mid = torch.empty(num_blocks, dtype=torch.int64, device=x.device)

        count_nonzero_kernel_1[(num_blocks,)](x, mid, numel, BLOCK_SIZE=BLOCK_SIZE)
        count_nonzero_kernel_2[(1,)](
            mid,
            out,
            num_blocks,
            BLOCK_SIZE=triton.next_power_of_2(num_blocks),
        )

        return out[0]
