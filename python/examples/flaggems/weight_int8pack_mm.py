import torch
import triton
import triton.language as tl


@triton.jit
def weight_int8pack_mm_kernel(
    A,
    B,
    Scales,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_scales,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in range(0, K, BLOCK_K):
        k_offs = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        mask_k = k_offs < K

        a = tl.load(
            A + ram[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            B + k_offs[:, None] * stride_bk + rbn[None, :] * stride_bn,
            mask=mask_k[:, None],
            other=0,
        )

        compute_dtype = C.dtype.element_ty
        a = a.to(compute_dtype)
        b = b.to(compute_dtype)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=True)

    scales_vec = tl.load(
        Scales + rn * stride_scales,
        mask=rn < N,
        other=0.0,
    ).to(tl.float32)
    acc = acc * scales_vec[None, :]

    acc = acc.to(C.dtype.element_ty)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    C = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)


def weight_int8pack_mm(A, B, scales):
    assert A.shape[1] == B.shape[1], (
        f"incompatible K dimensions: A.shape[1]={A.shape[1]}, B.shape[1]={B.shape[1]}"
    )
    assert B.shape[0] == scales.shape[0], (
        f"incompatible N dimensions: B.shape[0]={B.shape[0]}, scales.shape[0]={scales.shape[0]}"
    )
    assert B.dtype == torch.int8, f"B must be int8, got {B.dtype}"

    M, K = A.shape
    N = B.shape[0]

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)

    weight_int8pack_mm_kernel[grid](
        A,
        B,
        scales,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        scales.stride(0),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        GROUP_M=8,
    )

    return C
