import torch

import triton
import triton.language as tl
import benchmark


@triton.jit
def matmul_fp16_kernel(
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
    """FP16 matmul kernel: C = A x B  (all fp16, accumulate in fp16)."""
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

    # Accumulate in fp16 (matches IME vfmadot behaviour).
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b, out_dtype=tl.float16)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_fp16(a, b):
    """Wrapper: checks shapes, allocates output, launches fp16 kernel."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert a.dtype == torch.float16 and b.dtype == torch.float16

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    matmul_fp16_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        # Block sizes aligned with IME fp16 tile: 4x4x4
        BLOCK_SIZE_M=4,
        BLOCK_SIZE_N=4,
        BLOCK_SIZE_K=4,
        GROUP_SIZE_M=8,
    )
    return c


def test_matmul_fp16(device="cpu"):
    torch.manual_seed(42)
    M, K, N = 16, 16, 16
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)

    triton_output = matmul_fp16(a, b)
    # Reference: use float32 for torch.matmul then cast back
    torch_output = torch.matmul(a.float(), b.float()).half()

    # fp16 arithmetic is lossy; use a generous tolerance
    torch.testing.assert_close(triton_output, torch_output, atol=1e-1, rtol=1e-2)
    print(f"test_matmul_fp16 PASSED  (M={M}, K={K}, N={N})")


def test_matmul_fp16_boundary(device="cpu"):
    """Non-aligned dimensions to exercise the boundary-handling lowering path."""
    torch.manual_seed(7)
    M, K, N = 7, 6, 5
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)

    triton_output = matmul_fp16(a, b)
    torch_output = torch.matmul(a.float(), b.float()).half()

    torch.testing.assert_close(triton_output, torch_output, atol=1e-1, rtol=1e-2)
    print(f"test_matmul_fp16_boundary PASSED  (M={M}, K={K}, N={N})")


@benchmark.measure()
def bench_matmul_fp16(M, N, K, provider):
    a = torch.randn((M, K), device="cpu", dtype=torch.float16)
    b = torch.randn((K, N), device="cpu", dtype=torch.float16)
    if provider == "torch":
        torch.matmul(a, b)
    if provider == "triton":
        matmul_fp16(a, b)


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    test_matmul_fp16()
    test_matmul_fp16_boundary()
    for X in [16 * i for i in range(1, 6)]:
        for provider in ["torch", "triton"]:
            bench_matmul_fp16(X, X, X, provider)
