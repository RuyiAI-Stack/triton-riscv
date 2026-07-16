import os

import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
_WARNED_FALLBACK = False


def _warn_fallback_once(message: str):
    global _WARNED_FALLBACK
    if not _WARNED_FALLBACK:
        print(message)
        _WARNED_FALLBACK = True


@triton.jit
def matmul_kernel_aligned(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float32)

    c_ptrs = c_ptr + (
        offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    )
    tl.store(c_ptrs, c)


@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, activation=""):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    # CPU lowering materializes each Triton program's tiles in temporary
    # buffers.  Prefer the largest aligned tile so those buffers and the
    # launcher are reused for more arithmetic, while retaining the original
    # 32x64 minimum tile for smaller shapes.
    openmp_threads = max(1, int(os.getenv("TRITON_RISCV_OPENMP_THREADS", "1")))
    block_m_cap = max(32, 512 // openmp_threads)
    block_size_m = next(
        (
            block
            for block in (256, 128, 64, 32)
            if block <= block_m_cap and M % block == 0
        ),
        32,
    )
    m_programs = triton.cdiv(M, block_size_m)
    n_programs_needed = max(1, triton.cdiv(openmp_threads, m_programs))
    block_n_cap = max(64, N // n_programs_needed)
    block_size_n = next(
        (
            block
            for block in (512, 256, 128, 64)
            if block <= block_n_cap and N % block == 0
        ),
        64,
    )
    block_size_k = next(
        (block for block in (512, 256, 128, 64, 32, 16) if K % block == 0),
        16,
    )
    if openmp_threads == 1:
        # On the C920, a 32x32 output tile keeps the fixed-width RVV
        # accumulator working set cache-resident. A 64-wide K chunk amortizes
        # loop overhead without recreating the large temporary tiles used by
        # the throughput-oriented multi-thread configuration.
        block_size_m = next((block for block in (32, 16, 8) if M % block == 0), 8)
        block_size_n = next((block for block in (32, 16, 8) if N % block == 0), 8)
        block_size_k = next(
            (block for block in (64, 32, 16, 8) if K % block == 0), 8
        )
    block_size_m = int(os.getenv("TRITON_RISCV_MATMUL_BLOCK_M", block_size_m))
    block_size_n = int(os.getenv("TRITON_RISCV_MATMUL_BLOCK_N", block_size_n))
    block_size_k = int(os.getenv("TRITON_RISCV_MATMUL_BLOCK_K", block_size_k))
    aligned = M % block_size_m == 0 and N % block_size_n == 0 and K % block_size_k == 0
    if not aligned:
        _warn_fallback_once(
            "[matmul] Falling back to torch.matmul for non-aligned shapes to avoid masked staging."
        )
        result = torch.matmul(a, b)
        if activation == "leaky_relu":
            result = torch.where(result + 1 >= 0, result + 1, 0.01 * (result + 1))
        return result

    matmul_kernel_aligned[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        ACTIVATION=activation,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=8,
        allow_fp_reassoc=True,
    )
    return c


def bench_matmul(M, N, K):
    a = torch.randn((M, K), device="cpu", dtype=torch.float32)
    b = torch.randn((K, N), device="cpu", dtype=torch.float32)
    benchmark.compare_providers(
        f"bench_matmul(M={M}, N={N}, K={K})",
        {
            "torch": lambda: torch.matmul(a, b),
            "triton-riscv": lambda: matmul(a, b),
        },
        rtol=1e-3,
        atol=1e-2,
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    bench_matmul(512, 512, 512)
