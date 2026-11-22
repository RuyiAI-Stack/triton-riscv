"""Triton implementation of aten::inner.

Ported from upstream FlagGems ops/inner.py.

Contracts the last dimension of ``input`` with the last dimension of
``other``, keeping the leading dimensions of each operand per-side without
broadcasting:

* both 1-D      -> scalar dot product (split partials + tree reduce)
* one 0-D       -> outer-product scaling (no contraction dimension)
* otherwise     -> fused strided GEMM over the flattened operands
                   (``input @ other.transpose(-1, -2)``)

Porting notes (repo conventions):
  - @libentry stripped (plain @triton.jit); no torch device context
    manager; the kunlunxin tuning branches are not ported (fixed tile
    table from the non-XPU branch).
  - fp64 2-D inputs: the backend's fp64 ``tl.dot`` accumulates at ~fp32
    precision, so the IS_FP64 branch of ``_inner_gemm_kernel`` was
    restructured into a scalar-FMA outer-product loop (sequential over K),
    which reproduces full fp64 accumulation and matches aten CPU fp64
    bit-for-bit on this backend.
  - Every non-fp64 dtype (including int64, which has no ``tl.dot``
    lowering here) is cast to fp32 before the dot, per upstream's fp32
    accumulation contract; fp16/bf16 casts are exact.
  - The fp32 dot precision is fixed to ``ieee`` (upstream queried the
    backend for tf32x3 availability; the CPU backend has no tf32 path).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _inner_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_MASK: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    """Tiled GEMM computing ``inner(A, B)[m, n] = sum_k A[m, k] * B[n, k]``.

    ``A`` and ``B`` are both laid out as ``(rows, K)``, so ``B`` is the second
    operand already transposed; only its row stride is needed to stride ``K``.
    ``GROUP_M`` swizzles the tile order so neighbouring programs share ``A``/``B``
    panels through cache.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N

    if IS_FP64:
        # No trustworthy fp64 dot on this backend: accumulate with scalar
        # FMA over K, sequential like aten's fp64 matmul.
        a_vec = a_ptr + offs_m * stride_am
        b_vec = b_ptr + offs_n * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
        for kk in range(K):
            a = tl.load(a_vec + kk * stride_ak, mask=offs_m < M, other=0.0)
            b = tl.load(b_vec + kk * stride_bk, mask=offs_n < N, other=0.0)
            acc = tl.fma(a[:, None], b[None, :], acc)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            if HAS_MASK:
                k_remaining = K - k0
                a = tl.load(
                    a_ptrs,
                    mask=mask_m & (offs_k[None, :] < k_remaining),
                    other=0.0,
                ).to(tl.float32)
                b = tl.load(
                    b_ptrs,
                    mask=(offs_k[:, None] < k_remaining) & mask_n,
                    other=0.0,
                ).to(tl.float32)
            else:
                a = tl.load(a_ptrs).to(tl.float32)
                b = tl.load(b_ptrs).to(tl.float32)
            acc = tl.dot(a, b, acc, input_precision="ieee")
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if HAS_MASK:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_m & mask_n)
    else:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


@triton.jit
def _inner_dot_kernel(
    a_ptr,
    b_ptr,
    mid_ptr,
    K,
    BLOCK_SIZE: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    """Strided 1-D contraction writing one partial sum per program.

    ``IS_FP64`` selects the accumulation dtype: fp64 inputs accumulate in
    fp64 natively. Every other dtype accumulates in fp32 (matching upstream,
    which has no IS_FP64 flag in the ported file the tests pin).
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if IS_FP64:
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        for k0 in range(0, K, num_programs * BLOCK_SIZE):
            cur = k0 + offs
            mask = cur < K
            a = tl.load(a_ptr + cur, mask=mask, other=0.0)
            b = tl.load(b_ptr + cur, mask=mask, other=0.0)
            acc += a * b
    else:
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for k0 in range(0, K, num_programs * BLOCK_SIZE):
            cur = k0 + offs
            mask = cur < K
            a = tl.load(a_ptr + cur, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + cur, mask=mask, other=0.0).to(tl.float32)
            acc += a * b

    tl.store(mid_ptr + pid, tl.sum(acc, axis=0).to(mid_ptr.dtype.element_ty))


@triton.jit
def _inner_dot_reduce_kernel(mid_ptr, c_ptr, n_blocks, BLOCK_N: tl.constexpr):
    """Tree-reduce the per-program partial sums into the scalar output."""
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n_blocks
    partials = tl.load(mid_ptr + offs, mask=mask, other=0.0)
    tl.store(c_ptr, tl.sum(partials, axis=0).to(c_ptr.dtype.element_ty))


def _select_block_config(M, N, K, dtype):
    """Pick GEMM tile sizes (the non-XPU table)."""
    if M < 16 or N < 16:
        return 16, 16, 32
    return 64, 64, 32


def inner(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Compute the inner product of two tensors."""
    if input.dim() == 0 and other.dim() == 0:
        return (input * other).reshape(())

    if input.dtype != other.dtype:
        raise RuntimeError("both inputs should have same dtype")

    # A 0-D operand has no contraction axis, so inner degenerates to scaling the
    # other operand by the scalar.
    if input.dim() == 0:
        return (other * input).reshape(other.shape)
    if other.dim() == 0:
        return (input * other).reshape(input.shape)

    if input.size(-1) != other.size(-1):
        raise RuntimeError(
            "inner() the last dimension must match on both input tensors but "
            f"got shapes {list(input.shape)} and {list(other.shape)}"
        )

    out_dtype = input.dtype
    device = input.device

    if input.dim() == 1 and other.dim() == 1:
        is_fp64 = out_dtype == torch.float64
        # The per-program partials are fp32 (or fp64 for fp64 inputs) and the
        # reduce kernel writes the final scalar straight into the output dtype,
        # so the result needs no separate cast. `empty` is used because the
        # reduce kernel always stores the value, including for k == 0.
        mid_dtype = torch.float64 if is_fp64 else torch.float32
        out = torch.empty((), dtype=out_dtype, device=device)
        k = input.numel()
        if k == 0:
            return torch.zeros((), dtype=out_dtype, device=device)
        # 64 programs x 64 lanes saturates the reduction without paying for a
        # second launch; a single wide program serializes the loop.
        block_size = 64
        grid = (min((k + block_size - 1) // block_size, 64),)
        mid = torch.empty(grid[0], dtype=mid_dtype, device=device)
        _inner_dot_kernel[grid](
            input, other, mid, k, BLOCK_SIZE=block_size, IS_FP64=is_fp64
        )
        _inner_dot_reduce_kernel[(1,)](
            mid,
            out,
            grid[0],
            BLOCK_N=triton.next_power_of_2(grid[0]),
        )
        return out

    k = input.size(-1)
    out_shape = input.shape[:-1] + other.shape[:-1]
    # An empty contraction dimension is a well-defined all-zero reduction, and
    # `reshape(-1, 0)` is ambiguous, so short-circuit before flattening.
    if k == 0:
        return torch.zeros(out_shape, dtype=out_dtype, device=device)
    a2 = input.reshape(-1, k)
    b2 = other.reshape(-1, k)
    m = a2.shape[0]
    n = b2.shape[0]

    out = torch.empty((m, n), dtype=out_dtype, device=device)
    if m == 0 or n == 0:
        return out.view(out_shape)

    block_m, block_n, block_k = _select_block_config(m, n, k, out_dtype)
    grid = (((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n),)
    # Masks are redundant when every dimension divides its tile exactly.
    has_mask = not (m % block_m == 0 and n % block_n == 0 and k % block_k == 0)
    _inner_gemm_kernel[grid](
        a2,
        b2,
        out,
        m,
        n,
        k,
        a2.stride(0),
        a2.stride(1),
        b2.stride(0),
        b2.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        HAS_MASK=has_mask,
        IS_FP64=out_dtype == torch.float64,
    )
    return out.view(out_shape)
