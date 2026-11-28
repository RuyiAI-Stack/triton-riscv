import torch
import triton
import triton.language as tl


@triton.jit
def sparse_semi_structured_mm_kernel(
    A,
    Meta,
    B,
    C,
    M,
    N,
    K4,
    stride_am,
    stride_ak,
    stride_mm,
    stride_mn,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for sparse semi-structured matrix multiplication.

    A: (M, 4*K4) - dense matrix with 2:4 sparsity pattern
    Meta: (M, K4) - boolean mask indicating which 2 elements in each group of 4 are valid
    B: (4*K4, N) - dense matrix
    C: (M, N) - output matrix

    For each group of 4 elements in A's K dimension:
    - Meta[i, k] = True: use A[i, 4k] and A[i, 4k+1]
    - Meta[i, k] = False: use A[i, 4k+2] and A[i, 4k+3]
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Combine pid_m and pid_n into a single pid
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Calculate the actual K dimension
    K = K4 * 4

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_iters = tl.cdiv(K4, BLOCK_SIZE_K)

    # Loop over K dimension in blocks
    for k in range(num_iters):
        k_offset = k * BLOCK_SIZE_K

        # Load meta for this K block
        meta_ptrs = Meta + (
            offs_am[:, None] * stride_mm + (offs_k + k_offset)[None, :] * stride_mn
        )

        # Create a 2D mask: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        mask_2d = offs_am[:, None] < M
        mask_2d = mask_2d & ((offs_k + k_offset)[None, :] < K4)

        meta = tl.load(meta_ptrs, mask=mask_2d, other=0)

        # Load all 4 A elements for each K position
        a0_ptrs = A + (
            offs_am[:, None] * stride_am
            + (offs_k * 4 + k_offset * 4 + 0)[None, :] * stride_ak
        )
        mask_a = (offs_am[:, None] < M) & ((offs_k * 4 + k_offset * 4 + 0)[None, :] < K)
        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)

        a1_ptrs = A + (
            offs_am[:, None] * stride_am
            + (offs_k * 4 + k_offset * 4 + 1)[None, :] * stride_ak
        )
        mask_a = (offs_am[:, None] < M) & ((offs_k * 4 + k_offset * 4 + 1)[None, :] < K)
        a1 = tl.load(a1_ptrs, mask=mask_a, other=0.0)

        a2_ptrs = A + (
            offs_am[:, None] * stride_am
            + (offs_k * 4 + k_offset * 4 + 2)[None, :] * stride_ak
        )
        mask_a = (offs_am[:, None] < M) & ((offs_k * 4 + k_offset * 4 + 2)[None, :] < K)
        a2 = tl.load(a2_ptrs, mask=mask_a, other=0.0)

        a3_ptrs = A + (
            offs_am[:, None] * stride_am
            + (offs_k * 4 + k_offset * 4 + 3)[None, :] * stride_ak
        )
        mask_a = (offs_am[:, None] < M) & ((offs_k * 4 + k_offset * 4 + 3)[None, :] < K)
        a3 = tl.load(a3_ptrs, mask=mask_a, other=0.0)

        # Load B elements similarly
        b0_ptrs = B + (
            (offs_k * 4 + k_offset * 4 + 0)[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        mask_b = ((offs_k * 4 + k_offset * 4 + 0)[:, None] < K) & (offs_bn[None, :] < N)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)

        b1_ptrs = B + (
            (offs_k * 4 + k_offset * 4 + 1)[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        mask_b = ((offs_k * 4 + k_offset * 4 + 1)[:, None] < K) & (offs_bn[None, :] < N)
        b1 = tl.load(b1_ptrs, mask=mask_b, other=0.0)

        b2_ptrs = B + (
            (offs_k * 4 + k_offset * 4 + 2)[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        mask_b = ((offs_k * 4 + k_offset * 4 + 2)[:, None] < K) & (offs_bn[None, :] < N)
        b2 = tl.load(b2_ptrs, mask=mask_b, other=0.0)

        b3_ptrs = B + (
            (offs_k * 4 + k_offset * 4 + 3)[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        mask_b = ((offs_k * 4 + k_offset * 4 + 3)[:, None] < K) & (offs_bn[None, :] < N)
        b3 = tl.load(b3_ptrs, mask=mask_b, other=0.0)

        # For each (m, n), the result is sum over k of:
        #   if meta[m, k]: a0[m,k]*b0[k,n] + a1[m,k]*b1[k,n]
        #   else: a2[m,k]*b2[k,n] + a3[m,k]*b3[k,n]
        meta_exp = meta[:, :, None]  # (BM, BK, 1)

        prod00_exp = a0[:, :, None] * b0[None, :, :]
        prod01_exp = a1[:, :, None] * b1[None, :, :]
        prod02_exp = a2[:, :, None] * b2[None, :, :]
        prod03_exp = a3[:, :, None] * b3[None, :, :]

        selected = (prod00_exp + prod01_exp) * meta_exp + (prod02_exp + prod03_exp) * (
            1 - meta_exp
        )
        accumulator += tl.sum(selected, axis=1)

    # Store result
    c_ptrs = C + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _sparse_semi_structured_mm(mat1, mat1_meta, mat2, *, out_dtype=None):
    """Performs a sparse semi-structured matrix multiplication.

    Args:
        mat1: Dense matrix of shape (M, 4*K) with 2:4 sparsity pattern
        mat1_meta: Boolean mask of shape (M, K) indicating which elements are valid
        mat2: Dense matrix of shape (4*K, N)
        out_dtype: Optional output dtype

    Returns:
        Matrix of shape (M, N)
    """
    M = mat1.shape[0]
    K4 = mat1_meta.shape[1]
    N = mat2.shape[1]

    # Validate shapes
    assert mat1.shape == (
        M,
        4 * K4,
    ), f"Expected mat1 shape ({M}, {4 * K4}), got {mat1.shape}"
    assert mat2.shape == (
        4 * K4,
        N,
    ), f"Expected mat2 shape ({4 * K4}, {N}), got {mat2.shape}"
    assert mat1_meta.shape == (
        M,
        K4,
    ), f"Expected mat1_meta shape ({M}, {K4}), got {mat1_meta.shape}"

    # Determine output dtype
    if out_dtype is not None:
        output_dtype = out_dtype
    else:
        output_dtype = mat1.dtype

    out = torch.empty((M, N), device=mat1.device, dtype=output_dtype)

    grid = (triton.cdiv(M, 32) * triton.cdiv(N, 32),)

    sparse_semi_structured_mm_kernel[grid](
        mat1,
        mat1_meta,
        mat2,
        out,
        M,
        N,
        K4,
        mat1.stride(0),
        mat1.stride(1),
        mat1_meta.stride(0),
        mat1_meta.stride(1),
        mat2.stride(0),
        mat2.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
    )

    return out
