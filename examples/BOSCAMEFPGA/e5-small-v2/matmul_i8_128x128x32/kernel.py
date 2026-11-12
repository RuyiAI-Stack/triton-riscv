"""Standalone Triton kernel for FPGA case 'matmul_i8_128x128x32' (model 'e5-small-v2').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/e5-small-v2/matmul_i8_128x128x32
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
    # Exact INT32 dot product. AME requires K divisible by 64, while this
    # model has head_dim32. Keep the logical BK32 and use a scalar integer accumulation.
    row = tl.program_id(0)
    column = tl.program_id(1)
    value = tl.full((), 0, tl.int32)
    for lane in range(K):
        a = tl.load(A + row * K + lane).to(tl.int32)
        b = tl.load(B + column * K + lane).to(tl.int32)
        value = value + a * b
    tl.store(C + row * N + column, value)


KERNEL_CONFIG = {
    "family": "matmul_i8",
    "kernel": "matmul_i8",
    "arguments": [
        {"name": "A", "rank": 2, "dtype": "i8"},
        {"name": "B", "rank": 2, "dtype": "i8"},
        {"name": "C", "rank": 2, "dtype": "i32"},
    ],
    "shape": [128, 128, 32],
    "argument_shapes": [[128, 32], [128, 32], [128, 128]],
    "constexprs": {
        "M": 128,
        "VALID_M": 128,
        "N": 128,
        "K": 32,
        "BM": 1,
        "BN": 1,
        "BK": 32,
    },
    "grid": [128, 128, 1],
    "roles": ["model_pure_product"],
    "accumulation": "overwrite",
    "outputs": [2],
    "checked_elements": 16384,
    "metadata": {
        "model": "intfloat/e5-small-v2",
        "batch": 1,
        "sequence_length": 128,
        "target": "nr-fpga",
    },
    "implementation": "scalar integer accumulation fallback: logical K32 is below the AME K64 minimum",
}
