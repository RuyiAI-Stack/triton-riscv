"""Standalone Triton source for matmul_i8_overwrite_196x768x768."""

import triton
import triton.language as tl


@triton.jit
def matmul_i8_overwrite(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    VALID_M: tl.constexpr,
):
    """Signed INT8 C[M,N] = A[M,K] @ B[N,K]^T.

    M and VALID_M are the logical row count. Masked loads zero-pad each
    fixed-size AME tile internally; callers provide no padded DDR buffers.
    """
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.int32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        valid_a = (rows[:, None] < VALID_M) & (kk[None, :] < K)
        valid_b = (cols[None, :] < N) & (kk[:, None] < K)
        a = tl.load(A + rows[:, None] * K + kk[None, :], valid_a, other=0)
        b = tl.load(B + cols[None, :] * K + kk[:, None], valid_b, other=0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    valid = (rows[:, None] < VALID_M) & (cols[None, :] < N)
    address = C + rows[:, None] * N + cols[None, :]
    # The zero initial value is part of this compiled kernel, not a caller promise.
    tl.store(address, acc, valid)


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "matmul_i8_overwrite",
    "kernel": "matmul_i8_overwrite",
    "arguments": [
        {"name": "A", "rank": 2, "dtype": "i8"},
        {"name": "B", "rank": 2, "dtype": "i8"},
        {"name": "C", "rank": 2, "dtype": "i32"},
    ],
    "shape": [196, 768, 768],
    "constexprs": {"M": 196, "VALID_M": 196, "N": 768, "K": 768, "BM": 32, "BN": 64, "BK": 128},
    "grid": [7, 12, 1],
    "roles": ["model_pure_product"],
    "accumulation": "overwrite",
    "metadata": {"dtype": "i8 x i8 -> i32", "target": "nr-fpga", "checked_elements": 150528},
}
