import torch

import triton
import triton.language as tl
from triton.backends.riscv.driver import CPUDriver


# This implements einsum(qhmd,hmpd->qhmp).
@triton.jit
def einsum_qhmd_hmpd_to_qhmp_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    Q,
    H,
    M,
    P,
    D,
    strideAq,
    strideAh,
    strideAm,
    strideAd,
    strideBh,
    strideBm,
    strideBp,
    strideBd,
    strideCq,
    strideCh,
    strideCm,
    strideCp,
    BLOCK_P: tl.constexpr,
):
    """
    Triton kernel computing:
       C[q,h,m,p] = sum_{d=0..D-1} A[q,h,m,d] * B[h,m,p,d].

    We tile over P only. Each program instance computes one (q, h, m) row
    and a vector block along p.
    """

    pid_qhm = tl.program_id(axis=0)
    pid_p = tl.program_id(axis=1)
    p_idx = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    valid_qhm = pid_qhm < (Q * H * M)
    valid_p = p_idx < P

    # Decompose qhm = q * (H * M) + h * M + m -> (q, h, m)
    q_ = pid_qhm // (H * M)
    hm_ = pid_qhm % (H * M)
    h_ = hm_ // M
    m_ = hm_ % M

    baseA = q_ * strideAq + h_ * strideAh + m_ * strideAm
    baseB = h_ * strideBh + m_ * strideBm
    baseC = q_ * strideCq + h_ * strideCh + m_ * strideCm

    c_acc = tl.zeros((BLOCK_P,), dtype=tl.float32)
    for d_idx in range(D):
        a_val = tl.load(A_ptr + baseA + d_idx * strideAd, mask=valid_qhm, other=0.0)
        b_ptrs = B_ptr + baseB + d_idx * strideBd + p_idx * strideBp
        b_vals = tl.load(b_ptrs, mask=valid_p, other=0.0)
        c_acc += a_val * b_vals

    c_ptrs = C_ptr + baseC + p_idx * strideCp
    tl.store(c_ptrs, c_acc, mask=valid_qhm & valid_p)


def select_cpu_backend_compat():
    triton.runtime.driver.set_active(CPUDriver())


def einsum_qhmd_hmpd_to_qhmp(A, B, BLOCK_QHM=None, BLOCK_P=2):
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

    # Keep BLOCK_QHM for compatibility with older callers. The current kernel
    # lowers one (q, h, m) row per program instance, so BLOCK_QHM is ignored.
    _ = BLOCK_QHM

    grid = (Q * H * M, triton.cdiv(P, BLOCK_P))

    einsum_qhmd_hmpd_to_qhmp_kernel[grid](
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
        BLOCK_P,
    )

    return C


def test(device):
    Q, H, M, D = (1, 2, 2, 2)
    P = 4

    A = torch.randn((Q, H, M, D), dtype=torch.float32, device=device)
    B = torch.randn((H, M, P, D), dtype=torch.float32, device=device)

    # Reference
    C_ref = torch.einsum("qhmd,hmpd->qhmp", A, B)

    C_triton = einsum_qhmd_hmpd_to_qhmp(A, B)
    torch.testing.assert_close(C_triton, C_ref)
