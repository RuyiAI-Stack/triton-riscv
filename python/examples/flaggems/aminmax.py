import math

import torch
import triton
import triton.language as tl


@triton.jit
def get_dtype_max(dtype: tl.constexpr):
    """Get a value which is greater that all other values of that dtype"""
    # extract the tl.dtype from tl.constexpr so as to use its methods
    dtype_ = dtype.value
    if dtype_.is_floating():
        value: tl.constexpr = float("inf")
        return value
    if dtype_.is_int_signed():
        width: tl.constexpr = dtype_.int_bitwidth
        value: tl.constexpr = 2 ** (width - 1) - 1
        return value
    if dtype_.is_int_unsigned():
        width: tl.constexpr = dtype_.int_bitwidth
        value: tl.constexpr = 2**width - 1
        return value


@triton.jit
def get_dtype_min(dtype):
    """Get a value which is less that all other values of that dtype"""
    dtype_ = dtype.value  # tl.dtype
    if dtype_.is_floating():
        value: tl.constexpr = float("-inf")
        return value
    if dtype_.is_int_signed():
        width: tl.constexpr = dtype_.int_bitwidth
        value: tl.constexpr = -1 * 2 ** (width - 1)
        return value
    if dtype_.is_int_unsigned():
        value: tl.constexpr = 0
        return value


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
def aminmax_kernel_1(
    inp,
    min_out,
    max_out,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    min_fill = get_dtype_max(inp.type.element_ty)
    max_fill = get_dtype_min(inp.type.element_ty)
    min_val = tl.load(inp_ptrs, mask=mask, other=min_fill)
    max_val = tl.load(inp_ptrs, mask=mask, other=max_fill)

    min_val = tl.min(min_val)
    max_val = tl.max(max_val)

    min_ptr = min_out + pid
    max_ptr = max_out + pid
    tl.store(min_ptr, min_val)
    tl.store(max_ptr, max_val)


@triton.jit
def aminmax_kernel_2(
    min_inp, max_inp, min_out, max_out, mid_size, BLOCK_MID: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    min_ptrs = min_inp + offset
    max_ptrs = max_inp + offset
    mask = offset < mid_size
    min_fill = get_dtype_max(min_inp.type.element_ty)
    max_fill = get_dtype_min(max_inp.type.element_ty)
    min_val = tl.load(min_ptrs, mask=mask, other=min_fill)
    max_val = tl.load(max_ptrs, mask=mask, other=max_fill)

    min_val = tl.min(min_val)
    max_val = tl.max(max_val)

    tl.store(min_out, min_val)
    tl.store(max_out, max_val)


@triton.jit
def aminmax_kernel(
    inp,
    min_out,
    max_out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)
    max_value = get_dtype_max(dtype)

    # Map the program id to the row of inp it should compute.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    min_out = min_out + rows
    max_out = max_out + rows
    row_mask = rows < M

    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    _min = tl.full([BLOCK_M, BLOCK_N], value=max_value, dtype=acc_type)
    _max = tl.full([BLOCK_M, BLOCK_N], value=min_value, dtype=acc_type)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        a = tl.load(inp + cols, mask=mask, other=min_value)
        _min = tl.where(mask, tl.minimum(_min, a), _min)
        _max = tl.where(mask, tl.maximum(_max, a), _max)
    min_result = tl.min(_min, axis=1)[:, None]
    max_result = tl.max(_max, axis=1)[:, None]
    tl.store(min_out, min_result, row_mask)
    tl.store(max_out, max_result, row_mask)


def aminmax(inp, dim=None, keepdim=False, *, out=None):

    if dim is None:
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        min_mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        max_mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)

        if out is not None:
            min_out = out[0] if isinstance(out, tuple) else out
            max_out = out[1] if isinstance(out, tuple) else out
            if not keepdim:
                min_out = min_out.squeeze()
                max_out = max_out.squeeze()
        else:
            if not keepdim:
                min_out = torch.empty([], dtype=dtype, device=inp.device)
                max_out = torch.empty([], dtype=dtype, device=inp.device)
            else:
                shape = [1] * inp.dim()
                min_out = torch.empty(shape, dtype=dtype, device=inp.device)
                max_out = torch.empty(shape, dtype=dtype, device=inp.device)

        aminmax_kernel_1[(mid_size,)](
            inp,
            min_mid,
            max_mid,
            M,
            BLOCK_SIZE=block_size,
        )
        aminmax_kernel_2[(1,)](
            min_mid, max_mid, min_out, max_out, mid_size, BLOCK_MID=block_mid
        )
        return min_out, max_out
    else:
        if isinstance(dim, int):
            dim = [dim]

        for i in dim:
            assert i >= -inp.ndim and i < inp.ndim, "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        inp = dim_compress(inp, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        out_provided = out is not None
        if out_provided:
            min_out = out[0] if isinstance(out, tuple) else out
            max_out = out[1] if isinstance(out, tuple) else out
        else:
            min_out = torch.empty(shape, dtype=dtype, device=inp.device)
            max_out = torch.empty(shape, dtype=dtype, device=inp.device)

        BLOCK_M = 32
        BLOCK_N = 32
        grid = (triton.cdiv(M, BLOCK_M),)

        aminmax_kernel[grid](
            inp,
            min_out,
            max_out,
            M,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        if not keepdim and not out_provided:
            for d in sorted(dim, reverse=True):
                min_out = min_out.squeeze(dim=d)
                max_out = max_out.squeeze(dim=d)
        return min_out, max_out
