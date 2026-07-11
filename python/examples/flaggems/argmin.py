import math

import torch
import triton
import triton.language as tl


@triton.jit
def get_dtype_max(dtype: tl.constexpr):
    """Get a value which is greater that all other values of that dtype."""
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
def argmin_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    max_value = get_dtype_max(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=max_value)
    min_val, min_index = tl.min(inp_val, axis=0, return_indices=True)
    min_index = min_index + pid * BLOCK_SIZE
    mid_value_ptr = mid_value + pid
    min_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, min_val)
    tl.store(min_index_ptr, min_index)


@triton.jit
def argmin_kernel_2(
    mid_value,
    mid_index,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid_value.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    index_val = tl.argmin(mid_val, axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)


@triton.jit
def argmin_kernel_opt_k1(
    inp,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    max_val = get_dtype_max(dtype)

    min_vals = tl.full([BLOCK_M], dtype=acc_type, value=max_val)
    argmin_vals = tl.full([BLOCK_M], dtype=tl.int32, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        inp_vals = tl.load(inp + offset, mask=True)

        local_min, local_argmin = tl.min(
            inp_vals,
            1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        update = local_min < min_vals
        min_vals = tl.where(update, local_min, min_vals)
        candidate_index = (start_n + local_argmin).to(tl.int32)
        argmin_vals = tl.where(update, candidate_index, argmin_vals)

    out_ptr = out_index + m_offset
    tl.store(out_ptr, argmin_vals, mask=True)


@triton.jit
def argmin_split_K_kernel_merged(
    inp,
    out_index,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]  # (BLOCK_M, 1)
    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]  # (1, BLOCK_K)

    m_mask = m < M
    k_mask = k < K
    mk_mask = m_mask & k_mask

    compute_dtype = tl.float32 if dtype == tl.bfloat16 else dtype
    max_val = get_dtype_max(compute_dtype)

    global_min = tl.full((BLOCK_M, BLOCK_K), max_val, dtype=compute_dtype)
    global_argmin = tl.full((BLOCK_M, BLOCK_K), 0, dtype=tl.int32)

    for start_n in range(0, N, BLOCK_N):
        n = start_n + tl.arange(0, BLOCK_N)
        n_mask = n < N

        offset = m * N * K + n[:, None, None] * K + k[None, :, :]

        inp_vals = tl.load(
            inp + offset,
            mask=(m_mask & n_mask[:, None, None] & k_mask[None, :, :]),
            other=max_val,
        )
        inp_vals = inp_vals.to(compute_dtype)

        local_min, local_argmin = tl.min(
            inp_vals,
            0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        local_argmin = (local_argmin + start_n).to(tl.int32)

        mask = local_min < global_min
        global_min = tl.where(mask, local_min, global_min)
        global_argmin = tl.where(mask, local_argmin, global_argmin)

    out_offset = m * K + k  # (BLOCK_M, BLOCK_K)
    tl.store(out_index + out_offset, global_argmin, mask=mk_mask)


@triton.jit
def argmin_kernel(
    inp,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    max_value = get_dtype_max(dtype)
    min_values = tl.full([BLOCK_M], dtype=acc_type, value=max_value)
    # Loop-carried structured state currently supports i32 tensors, not i64.
    # Reduction indices are widened when stored to PyTorch's int64 output.
    argmin_values = tl.full([BLOCK_M], dtype=tl.int32, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=max_value)
        local_min, local_argmin = tl.min(
            inp_vals,
            1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        update = local_min < min_values
        min_values = tl.where(update, local_min, min_values)
        candidate_index = (start_n + local_argmin).to(tl.int32)
        argmin_values = tl.where(update, candidate_index, argmin_values)

    offset_index = m_offset * K + pid_k
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_index_ptrs, argmin_values, mask=mask1)


def argmin(inp, dim=None, keepdim=False, *, dtype=None):
    if dim is None:
        M = inp.numel()
        if dtype is None:
            dtype = inp.dtype
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid_value = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        mid_index = torch.empty(
            (mid_size,), dtype=torch.int64, device=inp.device
        )
        if keepdim:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=torch.int64, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int64, device=inp.device)

        argmin_kernel_1[(mid_size,)](
            inp,
            mid_value,
            mid_index,
            M,
            BLOCK_SIZE=block_size,
        )
        argmin_kernel_2[(1,)](
            mid_value,
            mid_index,
            out,
            mid_size,
            BLOCK_MID=block_mid,
        )
        return out
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        shape = inp.shape
        dim = dim % inp.ndim
        N = shape[dim]
        M = math.prod(shape[:dim])
        K = inp.numel() // M // N
        if not inp.dtype.is_floating_point:
            # Multi-result integer reductions are not yet lowered by
            # TritonToLinalg.  Float64 preserves the tested integer values and
            # keeps the reduction in the supported floating-point path.
            inp = inp.to(torch.float64)
        inp = inp.contiguous()

        shape_list = list(shape)
        shape_list[dim] = 1
        out_index = torch.empty(
            shape_list, dtype=torch.int64, device=inp.device
        )
        if not keepdim:
            out_index = torch.squeeze(out_index, dim)

        if K == 1 and inp.dtype != torch.int32 and inp.dtype != torch.int16:
            BLOCK_M = min(128, triton.next_power_of_2(M))
            BLOCK_N = min(4096, triton.next_power_of_2(N))
            grid = (triton.cdiv(M, BLOCK_M), K)
            argmin_kernel_opt_k1[grid](
                inp,
                out_index,
                M,
                N,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
        else:
            torch2triton_dtype = {
                torch.float16: tl.float16,
                torch.bfloat16: tl.bfloat16,
                torch.float32: tl.float32,
            }
            if (
                (N % 64 == 0 or N == 512)
                and (K % 32 == 0)
                and M % 8 == 0
                and inp.dtype != torch.int32
                and inp.dtype != torch.int16
            ):
                triton_dtype = torch2triton_dtype[inp.dtype]
                BLOCK_M = 8
                BLOCK_K = 32
                BLOCK_N = min(4096, triton.next_power_of_2(N))
                grid_for_split_K = (
                    triton.cdiv(M, BLOCK_M),
                    triton.cdiv(K, BLOCK_K),
                )
                argmin_split_K_kernel_merged[grid_for_split_K](
                    inp,
                    out_index,
                    M,
                    N,
                    K,
                    dtype=triton_dtype,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                )
            else:
                BLOCK_M = min(128, triton.next_power_of_2(M))
                BLOCK_N = min(4096, triton.next_power_of_2(N))
                grid = (triton.cdiv(M, BLOCK_M), K)
                argmin_kernel[grid](
                    inp,
                    out_index,
                    M,
                    N,
                    K,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                )

        return out_index
