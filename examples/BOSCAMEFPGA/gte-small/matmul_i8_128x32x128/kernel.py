"""Standalone Triton source for matmul_i8_197x768x768."""

import triton
import triton.language as tl


@triton.jit
def matmul_i8(
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
    "family": "matmul_i8",
    "kernel": "matmul_i8",
    "arguments": [
        {"name": "A", "rank": 2, "dtype": "i8"},
        {"name": "B", "rank": 2, "dtype": "i8"},
        {"name": "C", "rank": 2, "dtype": "i32"},
    ],
    "shape": [128, 32, 128],
    "argument_shapes": [[128, 128], [32, 128], [128, 32]],
    "constexprs": {
        "M": 128,
        "VALID_M": 128,
        "N": 32,
        "K": 128,
        "BM": 32,
        "BN": 32,
        "BK": 128,
    },
    "grid": [4, 1, 1],
    "roles": ["model_pure_product"],
    "accumulation": "overwrite",
    "outputs": [2],
    "checked_elements": 4096,
    "metadata": {
        "model": "thenlper/gte-small",
        "batch": 1,
        "sequence_length": 128,
        "target": "nr-fpga",
    },
}
