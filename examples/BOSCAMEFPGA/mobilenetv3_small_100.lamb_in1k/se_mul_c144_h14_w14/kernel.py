"""
Standalone Triton kernel for FPGA case 'se_mul_c144_h14_w14' (model
'mobilenetv3_small_100.lamb_in1k').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in
examples/BOSCAMEFPGA/mobilenetv3_small_100.lamb_in1k/se_mul_c144_h14_w14 before
linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def se_mul(X, Scale, Out, COUNT: tl.constexpr, SPATIAL: tl.constexpr):
    """One scalar program per element. Channel scale is a scalar load."""
    index = tl.program_id(0)
    channel = index // SPATIAL
    x = tl.load(X + index)
    scale = tl.load(Scale + channel)
    tl.store(Out + index, x * scale)


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "se_mul",
    "kernel": "se_mul",
    "arguments": [
        {"name": "X", "rank": 1, "dtype": "f32"},
        {"name": "Scale", "rank": 1, "dtype": "f32"},
        {"name": "Out", "rank": 1, "dtype": "f32"},
    ],
    "shape": [1, 144, 14, 14],
    "constexprs": {"COUNT": 28224, "SPATIAL": 196},
    "grid": [28224, 1, 1],
    "roles": [],
}
