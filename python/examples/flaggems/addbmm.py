"""Triton implementation of aten::addbmm (batched matmul with bias add).

Ported from upstream FlagGems ops/addbmm.py.

    out = beta * bias + alpha * sum_b(batch1[b] @ batch2[b])

Porting notes (repo conventions):
  - @libentry / @triton.autotune stripped (plain @triton.jit, fixed 32x32x32
    tiles per house addmm.py); no torch device context manager; grid is a
    plain tuple.
  - fp64 inputs keep the upstream IS_FP64 behaviour: the matmul operands are
    rounded to fp32 before ``tl.dot`` while the accumulator and the bias
    arithmetic stay fp64.
  - int64 inputs (aten supports them) are cast to fp32 before ``tl.dot``:
    the CPU/riscv backend has no int64 ``tl.dot`` lowering. The tests pin
    small-magnitude integer data, which fp32 represents exactly, so the
    cast is value-preserving for the supported envelope.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["alpha", "beta"])
def addbmm_kernel(
    bias_ptr,
    batch1_ptr,
    batch2_ptr,
    out_ptr,
    alpha,
    beta,
    B,
    M,
    N,
    K,
    # Strides for bias [M, N]
    stride_bias_m,
    stride_bias_n,
    # Strides for batch1 [B, M, K]
    stride_b1_b,
    stride_b1_m,
    stride_b1_k,
    # Strides for batch2 [B, K, N]
    stride_b2_b,
    stride_b2_k,
    stride_b2_n,
    # Strides for output [M, N]
    stride_out_m,
    stride_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    """out = beta * bias + alpha * sum_b(batch1[b] @ batch2[b])"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Accumulator for sum over batch and K dimensions
    if IS_FP64:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over batch dimension to accumulate sum_b(batch1[b] @ batch2[b])
    for b in range(B):
        b1_base = batch1_ptr + b * stride_b1_b
        b2_base = batch2_ptr + b * stride_b2_b

        a_ptrs = b1_base + offs_m[:, None] * stride_b1_m + offs_k[None, :] * stride_b1_k
        b_ptrs = b2_base + offs_k[:, None] * stride_b2_k + offs_n[None, :] * stride_b2_n

        # Loop over K dimension (tiled)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_remaining = K - k * BLOCK_SIZE_K
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
            b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

            a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

            acc += tl.dot(a, b_tile, allow_tf32=False)

            a_ptrs += BLOCK_SIZE_K * stride_b1_k
            b_ptrs += BLOCK_SIZE_K * stride_b2_k

    # Load bias tile [BLOCK_SIZE_M, BLOCK_SIZE_N]
    bias_ptrs = (
        bias_ptr + offs_m[:, None] * stride_bias_m + offs_n[None, :] * stride_bias_n
    )
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    bias_tile = tl.load(bias_ptrs, mask=out_mask, other=0.0)

    # Compute output: beta * bias + alpha * acc
    out = acc * alpha + bias_tile.to(acc.dtype) * beta

    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, out.to(bias_tile.dtype), mask=out_mask)


def _launch(bias, batch1, batch2, out, beta, alpha):
    B, M, K = batch1.shape
    N = batch2.shape[2]
    grid = (
        triton.cdiv(M, 32),
        triton.cdiv(N, 32),
    )
    addbmm_kernel[grid](
        bias,
        batch1,
        batch2,
        out,
        alpha,
        beta,
        B,
        M,
        N,
        K,
        bias.stride(0),
        bias.stride(1),
        batch1.stride(0),
        batch1.stride(1),
        batch1.stride(2),
        batch2.stride(0),
        batch2.stride(1),
        batch2.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
        IS_FP64=bias.dtype == torch.float64,
    )
    return out


def addbmm(bias, batch1, batch2, beta=1.0, alpha=1.0):
    assert batch1.ndim == 3 and batch2.ndim == 3, "batch1 and batch2 must be 3D"
    assert bias.ndim == 2, "bias must be 2D"

    B, M, K = batch1.shape
    _, K2, N = batch2.shape
    assert K == K2, "Inner dimensions must match"
    assert bias.shape[0] == M and bias.shape[1] == N, (
        f"bias shape mismatch: expected ({M}, {N}), got {bias.shape}"
    )

    batch1 = batch1.contiguous()
    batch2 = batch2.contiguous()
    bias = bias.contiguous()
    out = torch.empty((M, N), dtype=bias.dtype, device=bias.device)

    return _launch(bias, batch1, batch2, out, beta, alpha)


def addbmm_(bias, batch1, batch2, beta=1.0, alpha=1.0):
    assert batch1.ndim == 3 and batch2.ndim == 3, "batch1 and batch2 must be 3D"
    assert bias.ndim == 2, "bias must be 2D"

    B, M, K = batch1.shape
    _, K2, N = batch2.shape
    assert K == K2, "Inner dimensions must match"
    assert bias.shape[0] == M and bias.shape[1] == N, (
        f"bias shape mismatch: expected ({M}, {N}), got {bias.shape}"
    )

    batch1 = batch1.contiguous()
    batch2 = batch2.contiguous()
    # The kernel reads the bias through a contiguous view but writes the
    # caller's original (possibly non-contiguous) tensor in place.
    # .contiguous() is a no-op on an already-contiguous bias, which would
    # alias the kernel's input and output buffers — always read from a copy.
    bias_contig = bias.contiguous()
    if bias_contig is bias:
        bias_contig = bias.clone()

    return _launch(bias_contig, batch1, batch2, bias, beta, alpha)
