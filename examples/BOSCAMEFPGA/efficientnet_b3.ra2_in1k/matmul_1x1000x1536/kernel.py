"""Standalone Triton source for matmul_1x1000x1536."""

import triton
import triton.language as tl

@triton.jit
def matmul(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    INTEGER: tl.constexpr,
):
    """Signed INT8 C[M, N] += A[M, K] @ B[N, K]^T."""
    tl.static_assert(INTEGER == True)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        a = tl.load(
            A + rows[:, None] * K + kk[None, :],
            (rows[:, None] < M) & (kk[None, :] < K),
            other=0,
        )
        b = tl.load(
            B + cols[None, :] * K + kk[:, None],
            (cols[None, :] < N) & (kk[:, None] < K),
            other=0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    address = C + rows[:, None] * N + cols[None, :]
    previous = tl.load(address, valid, other=0)
    tl.store(address, acc + previous, valid)


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "matmul_i8",
    "kernel": "matmul",
    "arguments": [
        {
            "name": "A",
            "rank": 2,
            "dtype": "i8",
        },
        {
            "name": "B",
            "rank": 2,
            "dtype": "i8",
        },
        {
            "name": "C",
            "rank": 2,
            "dtype": "i32",
        },
    ],
    "shape": [1, 1000, 1536],
    "constexprs": {
        "M": 1,
        "N": 1000,
        "K": 1536,
        "BM": 1,
        "BN": 16,
        "BK": 64,
        "INTEGER": True,
    },
    "grid": [1, 63, 1],
    "roles": [],
    "metadata": {
        "dtype": "efficientnet-b3",
        "model": "timm/efficientnet_b3.ra2_in1k",
    },
}
