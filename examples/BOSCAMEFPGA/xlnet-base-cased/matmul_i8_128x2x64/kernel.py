"""Standalone Triton kernel for FPGA case 'matmul_i8_128x2x64' (model 'xlnet-base-cased').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/xlnet-base-cased/matmul_i8_128x2x64
before linking launch.c.
"""

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
    """Exact signed INT8 scalar row-dot for narrow N=2 without AME padding."""
    tl.static_assert(BM == 1 and BN == 1 and K % BK == 0)
    row = tl.program_id(0)
    column = tl.program_id(1)
    accumulator = tl.full((), 0, tl.int32)
    for block in range(tl.cdiv(K, BK)):
        for lane in range(BK):
            k = block * BK + lane
            a = tl.load(A + row * K + k).to(tl.int32)
            b = tl.load(B + column * K + k).to(tl.int32)
            accumulator += a * b
    tl.store(C + row * N + column, accumulator)


KERNEL_CONFIG = {
    "family": "matmul_i8",
    "kernel": "matmul_i8",
    "arguments": [
        {"name": "A", "rank": 2, "dtype": "i8"},
        {"name": "B", "rank": 2, "dtype": "i8"},
        {"name": "C", "rank": 2, "dtype": "i32"},
    ],
    "shape": [128, 2, 64],
    "constexprs": {
        "M": 128,
        "VALID_M": 128,
        "N": 2,
        "K": 64,
        "BM": 1,
        "BN": 1,
        "BK": 64,
    },
    "grid": [128, 2, 1],
    "roles": ["model_pure_product"],
    "accumulation": "overwrite",
    "metadata": {
        "dtype": "i8 x i8 -> i32",
        "target": "nr-fpga",
        "checked_elements": 151296,
        "model": "xlnet/xlnet-base-cased",
    },
    "implementation": "Exact scalar int32 row-dot fallback for narrow segment N=2; no oversized logical tile",
}
