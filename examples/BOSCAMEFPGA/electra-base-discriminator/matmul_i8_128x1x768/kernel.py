"""Standalone Triton source for matmul_i8_128x1x768."""

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
    """Exact INT8 row-dot fallback for N=1 without an oversized AME tile."""
    tl.static_assert(BM == 1 and BN == 1 and K % BK == 0)
    row = tl.program_id(0)
    column = tl.program_id(1)
    # Scalar integer loads avoid AME's minimum N tile and the NR target's
    # unsupported vector i8 widening. Blocking never exceeds the logical K.
    accumulator = tl.full((), 0, tl.int32)
    for block in range(tl.cdiv(K, BK)):
        for lane in range(BK):
            k = block * BK + lane
            a = tl.load(A + row * K + k).to(tl.int32)
            b = tl.load(B + column * K + k).to(tl.int32)
            accumulator += a * b
    value = accumulator
    tl.store(C + row * N + column, value)


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "matmul_i8",
    "kernel": "matmul_i8",
    "arguments": [
        {"name": "A", "rank": 2, "dtype": "i8"},
        {"name": "B", "rank": 2, "dtype": "i8"},
        {"name": "C", "rank": 2, "dtype": "i32"},
    ],
    "shape": [128, 1, 768],
    "constexprs": {
        "M": 128,
        "VALID_M": 128,
        "N": 1,
        "K": 768,
        "BM": 1,
        "BN": 1,
        "BK": 128,
    },
    "grid": [128, 1, 1],
    "roles": ["model_pure_product"],
    "accumulation": "overwrite",
    "metadata": {
        "dtype": "i8 x i8 -> i32",
        "target": "nr-fpga",
        "checked_elements": 128,
        "model": "google/electra-base-discriminator",
    },
}
