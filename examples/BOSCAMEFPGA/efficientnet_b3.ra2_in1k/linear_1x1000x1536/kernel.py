"""Standalone Triton source for linear_1x1000x1536."""

import triton
import triton.language as tl

@triton.jit
def linear(
    A, W, Bias, Out,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """Out[M, N] = A[M, K] @ W[N, K]^T + Bias[N]."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        a = tl.load(
            A + rows[:, None] * K + kk[None, :],
            (rows[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            W + cols[None, :] * K + kk[:, None],
            (cols[None, :] < N) & (kk[:, None] < K),
            other=0.0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    acc += tl.load(Bias + cols, cols < N, other=0.0)[None, :]
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(Out + rows[:, None] * N + cols[None, :], acc, valid)


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "linear_f32",
    "kernel": "linear",
    "arguments": [
        {
            "name": "A",
            "rank": 2,
            "dtype": "f32",
        },
        {
            "name": "W",
            "rank": 2,
            "dtype": "f32",
        },
        {
            "name": "Bias",
            "rank": 1,
            "dtype": "f32",
        },
        {
            "name": "Out",
            "rank": 2,
            "dtype": "f32",
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
    },
    "grid": [1, 63, 1],
    "roles": [],
    "metadata": {
        "dtype": "efficientnet-b3",
        "model": "timm/efficientnet_b3.ra2_in1k",
    },
}
