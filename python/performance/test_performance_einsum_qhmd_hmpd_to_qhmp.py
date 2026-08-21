import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


# This implements einsum(qhmd,hmpd->qhmp).
@triton.jit
def einsum_qhmd_hmpd_to_qhmp_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    Q: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    strideAq: tl.constexpr,
    strideAh: tl.constexpr,
    strideAm: tl.constexpr,
    strideAd: tl.constexpr,
    strideBh: tl.constexpr,
    strideBm: tl.constexpr,
    strideBp: tl.constexpr,
    strideBd: tl.constexpr,
    strideCq: tl.constexpr,
    strideCh: tl.constexpr,
    strideCm: tl.constexpr,
    strideCp: tl.constexpr,
):
    """
    Triton kernel computing:
       C[q,h,m,p] = sum_{d=0..D-1} A[q,h,m,d] * B[h,m,p,d].

    Use structured scalar loops on CPU. These benchmark shapes are small, so
    2x2 tensor tiles spend substantially more time in staging and grid dispatch
    than in the contraction itself.
    """
    for q in tl.range(0, Q):
        for h in tl.range(0, H):
            for m in tl.range(0, M):
                a_base = q * strideAq + h * strideAh + m * strideAm
                b_base = h * strideBh + m * strideBm
                c_base = q * strideCq + h * strideCh + m * strideCm
                for p in tl.range(0, P):
                    accumulator = 0.0
                    for d in tl.range(0, D):
                        a = tl.load(A_ptr + a_base + d * strideAd)
                        b = tl.load(B_ptr + b_base + p * strideBp + d * strideBd)
                        accumulator += a * b
                    tl.store(C_ptr + c_base + p * strideCp, accumulator)


def einsum_qhmd_hmpd_to_qhmp(A, B, BLOCK_QHM=2, BLOCK_P=2):
    """
    A: [Q,H,M,D], B: [H,M,P,D]
    => C: [Q,H,M,P] with sum_{d=0..D-1} A[q,h,m,d]*B[h,m,p,d].
    """

    # Assertions to make sure we got shapes compatible with the fixed einsum implementation
    Q, H, M, D = A.shape
    assert B.shape[0] == H, f"B's H={B.shape[0]} != {H}"
    assert B.shape[1] == M, f"B's M={B.shape[1]} != {M}"
    P = B.shape[2]
    assert B.shape[3] == D, f"B's D={B.shape[3]} != {D}"

    C = torch.empty((Q, H, M, P), device=A.device, dtype=A.dtype)

    einsum_qhmd_hmpd_to_qhmp_kernel[(1,)](
        A,
        B,
        C,
        Q,
        H,
        M,
        P,
        D,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        A.stride(3),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        B.stride(3),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        C.stride(3),
    )

    return C


def bench_einsum_qhmd_hmpd_to_qhmp(Q, H, M, P, D):
    A = torch.rand((Q, H, M, D), device="cpu", dtype=torch.float32)
    B = torch.rand((H, M, P, D), device="cpu", dtype=torch.float32)
    benchmark.compare_providers(
        f"bench_einsum_qhmd_hmpd_to_qhmp(Q={Q}, H={H}, M={M}, P={P}, D={D})",
        {
            "torch": lambda: torch.einsum("qhmd,hmpd->qhmp", A, B),
            "triton-riscv": lambda: einsum_qhmd_hmpd_to_qhmp(A, B),
        },
        rtol=1e-4,
        atol=1e-4,
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for Q, H, M, P, D in [
        (1 * 2, 2 * 2, 2 * 2, 4 * 2, 2 * 2),
        (2 * 2, 4 * 2, 4 * 2, 2 * 2, 4 * 2),
        (4 * 2, 2 * 2, 2 * 2, 16 * 2, 2 * 2),
    ]:
        bench_einsum_qhmd_hmpd_to_qhmp(Q, H, M, P, D)
