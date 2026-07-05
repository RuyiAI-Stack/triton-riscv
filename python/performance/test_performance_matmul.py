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
    M,
    N,
    K,
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

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        accumulator += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float32)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c)


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

    block_size_m = 32
    block_size_n = 64
    block_size_k = 16
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
