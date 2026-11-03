"""Standalone Triton source for conv2d_18x18_576x96_k1_s1 (i8 x i8 -> i32).

1x1 NCHW conv is lowered as a packed GEMM:
  X[H*W, C] @ Weight[OC, C]^T -> Out[H*W, OC]  (i8 x i8 -> i32)
Bias is applied in launch.c (contract rejects bias broadcast into the tile).
"""

import triton
import triton.language as tl


@triton.jit
def conv2d_nchw(
    X,
    Weight,
    Bias,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    INTEGER: tl.constexpr,
):
    """Signed INT8 Out[M, N] = X[M, K] @ Weight[N, K]^T (Bias added in launch)."""
    tl.static_assert(INTEGER == True)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        a = tl.load(
            X + rows[:, None] * K + kk[None, :],
            (rows[:, None] < M) & (kk[None, :] < K),
            other=0,
        )
        b = tl.load(
            Weight + cols[None, :] * K + kk[:, None],
            (cols[None, :] < N) & (kk[:, None] < K),
            other=0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    # Keep Bias in the ABI; value applied in launch.c.
    _ = tl.load(Bias + cols, cols < N, other=0)
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(Out + rows[:, None] * N + cols[None, :], acc, valid)


# Fixed specialization; cases.py reads this literal without importing Triton.
# shape mirrors NCHW [1,C,H,W]; GEMM uses M=H*W, N=OC, K=C.
KERNEL_CONFIG = {
    "family": "conv2d_i8",
    "kernel": "conv2d_nchw",
    "arguments": [
        {"name": "X", "rank": 2, "dtype": "i8"},
        {"name": "Weight", "rank": 2, "dtype": "i8"},
        {"name": "Bias", "rank": 1, "dtype": "i32"},
        {"name": "Out", "rank": 2, "dtype": "i32"},
    ],
    "shape": [1, 576, 18, 18],
    "constexprs": {
        "M": 324,
        "N": 96,
        "K": 576,
        "BM": 16,
        "BN": 16,
        "BK": 64,
        "INTEGER": True,
    },
    "grid": [21, 6, 1],
    "roles": [],
    "metadata": {
        "dtype": "i8 x i8 -> i32",
        "layout": "packed GEMM X[H*W,C] @ W[OC,C]^T",
        "nchw": [1, 576, 18, 18],
        "model": "timm/efficientnet_b3.ra2_in1k",
    },
}
