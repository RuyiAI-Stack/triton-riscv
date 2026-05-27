import torch
import numpy as np

import triton
import triton.language as tl
import benchmark


@triton.jit
def matmul_int8_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Int8 matmul kernel: C (i32) = A (i8) x B (i8).

    Accumulates in int32 to avoid overflow, matching the IME vmadot
    instruction's widening-multiply-accumulate semantics.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulate in int32 (matches IME vmadot widening MAC behaviour).
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        accumulator += tl.dot(a, b, out_dtype=tl.int32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator  # already int32

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_int8(a, b):
    """Wrapper: checks shapes, allocates int32 output, launches int8 kernel."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert a.dtype == torch.int8 and b.dtype == torch.int8

    M, K = a.shape
    K, N = b.shape
    # Output is int32 (widening accumulation)
    c = torch.zeros((M, N), device=a.device, dtype=torch.int32)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    matmul_int8_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        # Block sizes aligned with IME int8 tile: TILE_M=4, TILE_K=8, TILE_N=4
        BLOCK_SIZE_M=4,
        BLOCK_SIZE_N=4,
        BLOCK_SIZE_K=8,
        GROUP_SIZE_M=8,
    )
    return c


def test_matmul_int8(device="cpu"):
    torch.manual_seed(42)
    M, K, N = 16, 16, 16
    # Use small values to avoid int8 overflow in the reference matmul
    a = torch.randint(-16, 16, (M, K), dtype=torch.int8, device=device)
    b = torch.randint(-16, 16, (K, N), dtype=torch.int8, device=device)

    triton_output = matmul_int8(a, b)

    # Reference: numpy int32 matmul (torch.matmul doesn't support i8 on CPU)
    a_np = a.numpy().astype(np.int32)
    b_np = b.numpy().astype(np.int32)
    torch_output = torch.from_numpy(a_np @ b_np)

    torch.testing.assert_close(triton_output, torch_output, atol=0, rtol=0)
    print(f"test_matmul_int8 PASSED  (M={M}, K={K}, N={N})")


def test_matmul_int8_boundary(device="cpu"):
    """Non-aligned dimensions to exercise boundary-handling lowering path."""
    torch.manual_seed(7)
    M, K, N = 7, 9, 5   # K=9 is not a multiple of BLOCK_SIZE_K=8
    a = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=device)
    b = torch.randint(-8, 8, (K, N), dtype=torch.int8, device=device)

    triton_output = matmul_int8(a, b)
    a_np = a.numpy().astype(np.int32)
    b_np = b.numpy().astype(np.int32)
    torch_output = torch.from_numpy(a_np @ b_np)

    torch.testing.assert_close(triton_output, torch_output, atol=0, rtol=0)
    print(f"test_matmul_int8_boundary PASSED  (M={M}, K={K}, N={N})")


@benchmark.measure()
def bench_matmul_int8(M, N, K, provider):
    a = torch.randint(-16, 16, (M, K), dtype=torch.int8, device="cpu")
    b = torch.randint(-16, 16, (K, N), dtype=torch.int8, device="cpu")
    if provider == "triton":
        matmul_int8(a, b)


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    test_matmul_int8()
    test_matmul_int8_boundary()
    for X in [16 * i for i in range(1, 6)]:
        bench_matmul_int8(X, X, X, "triton")
