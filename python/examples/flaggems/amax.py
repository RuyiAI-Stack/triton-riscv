import math

import torch
import triton
import triton.language as tl


@triton.jit
def get_dtype_min(dtype):
    """Get a value which is less that all other values of that dtype."""
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


@triton.jit
def amax_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    min_value = get_dtype_min(inp.dtype.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
    amax_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, amax_val)


@triton.jit
def amax_kernel_2(
    mid,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid.dtype.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    amax_val = tl.max(mid_val)
    tl.store(out, amax_val)


@triton.jit
def amax_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = inp.dtype.element_ty
    min_value = get_dtype_min(dtype)

    # Map the program id to the row of inp it should compute.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp_ptrs = inp + rows * N
    out_ptrs = out + rows
    row_mask = rows < M

    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    _all = tl.full([BLOCK_M, BLOCK_N], value=min_value, dtype=acc_type)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        a = tl.load(inp_ptrs + cols, mask=mask, other=min_value)
        _all = tl.maximum(_all, a)

    all = tl.max(_all, axis=1)[:, None]
    tl.store(out_ptrs, all, mask=row_mask)


def _dim_compress(inp, dims):
    if not isinstance(dims, (list, tuple)):
        dims = [dims]

    dims = [d % inp.ndim for d in dims]
    other_dims = [i for i in range(inp.ndim) if i not in dims]
    perm = other_dims + dims

    inp_perm = inp.permute(perm).contiguous()

    N = 1
    for d in dims:
        N *= inp.shape[d]
    M = inp.numel() // N

    return inp_perm.view(M, N)


def amax(inp, dim=None, keepdim=False):
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        amax_kernel_1[(mid_size,)](
            inp,
            mid,
            M,
            BLOCK_SIZE=block_size,
        )
        amax_kernel_2[(1,)](
            mid, out, mid_size, BLOCK_MID=block_mid
        )  # max block size is 128k, so mid does not requires int64 index
        return out
    else:
        if isinstance(dim, int):
            dim = [dim]
        for i in dim:
            assert i >= -inp.ndim and i < inp.ndim, "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        inp = _dim_compress(inp, dim)

        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        out = torch.empty(shape, dtype=dtype, device=inp.device)
        BLOCK_M = 32
        BLOCK_N = 32
        grid = (triton.cdiv(M, BLOCK_M),)

        amax_kernel[grid](
            inp,
            out,
            M,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        if not keepdim:
            out = out.squeeze(dim=dim)
        return out
