"""
Standalone Triton kernel for FPGA case 'hardswish_count14112' (model
'mobilenetv3_small_100.lamb_in1k').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in
examples/BOSCAMEFPGA/mobilenetv3_small_100.lamb_in1k/hardswish_count14112 before
linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def hardswish(X, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    x = tl.load(X + index, valid, other=0.0)
    shifted = x + 3.0
    lower = tl.maximum(shifted, 0.0)
    clamped = tl.minimum(lower, 6.0)
    # Keep aten's FP32 multiply-then-divide order, including overflow behavior.
    tl.store(Out + index, tl.div_rn(x * clamped, 6.0), valid)


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "hardswish",
    "kernel": "hardswish",
    "arguments": [
        {"name": "X", "rank": 1, "dtype": "f32"},
        {"name": "Out", "rank": 1, "dtype": "f32"},
    ],
    "shape": [14112],
    "constexprs": {"COUNT": 14112, "BLOCK": 128},
    "grid": [111, 1, 1],
    "roles": [],
}
