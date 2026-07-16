import torch
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    A,
    B,
    Out,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    pid_b = tl.program_id(2)
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    Out += pid_b * stride_ob

    pidx = tl.program_id(0)
    pidy = tl.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        gridx = tl.num_programs(0)
        gridy = tl.num_programs(1)
        pid = pidx + pidy * gridx
        num_CTA_per_group = gridy * GROUP_M
        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    o_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    num_iters = tl.cdiv(K, TILE_K)
    if IS_FP64:
        o = tl.zeros((TILE_M, TILE_N), dtype=tl.float64)
    else:
        o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for _ in range(num_iters):
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            if DIVISIBLE_N:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        else:
            mask_k = offs_k < K
            if DIVISIBLE_M:
                a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
            if DIVISIBLE_N:
                b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            else:
                b = tl.load(
                    b_ptrs,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                )

        offs_k += TILE_K
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

        o += tl.dot(a, b, allow_tf32=False)

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M and not DIVISIBLE_N:
        mask_c = mask_n[None, :]
    elif not DIVISIBLE_M and DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(o_ptrs, o, mask_c)


def bmm(A, B):
    assert A.shape[0] == B.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    batch, M, K = A.shape
    _, _, N = B.shape
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    TILE_M = 128
    TILE_N = 128
    TILE_K = 32
    GROUP_M = 8
    grid = (
        triton.cdiv(M, TILE_M),
        triton.cdiv(N, TILE_N),
        batch,
    )
    bmm_kernel[grid](
        A,
        B,
        out,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        GROUP_M=GROUP_M,
        DIVISIBLE_M=(M % TILE_M == 0),
        DIVISIBLE_N=(N % TILE_N == 0),
        DIVISIBLE_K=(K % TILE_K == 0),
        IS_FP64=(A.dtype == torch.float64),
    )
    return out


def bmm_out(A, B, out):
    assert A.shape[0] == B.shape[0] == out.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    batch, M, K = A.shape
    _, _, N = B.shape

    TILE_M = 128
    TILE_N = 128
    TILE_K = 32
    GROUP_M = 8
    grid = (
        triton.cdiv(M, TILE_M),
        triton.cdiv(N, TILE_N),
        batch,
    )
    bmm_kernel[grid](
        A,
        B,
        out,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        GROUP_M=GROUP_M,
        DIVISIBLE_M=(M % TILE_M == 0),
        DIVISIBLE_N=(N % TILE_N == 0),
        DIVISIBLE_K=(K % TILE_K == 0),
        IS_FP64=(A.dtype == torch.float64),
    )
    return out
