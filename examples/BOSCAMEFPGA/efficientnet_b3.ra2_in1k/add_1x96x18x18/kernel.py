"""
Standalone Triton kernel for FPGA case 'add_1x96x18x18' (model
'efficientnet_b3.ra2_in1k').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in
examples/BOSCAMEFPGA/efficientnet_b3.ra2_in1k/add_1x96x18x18 before linking
launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def binary(X, Y, Out, COUNT: tl.constexpr, MULTIPLY: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    x = tl.load(X + index, valid, other=0.0)
    y = tl.load(Y + index, valid, other=0.0)
    result = x * y if MULTIPLY else x + y
    tl.store(Out + index, result, valid)


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "add",
    "kernel": "binary",
    "arguments": [
        {
            "name": "X",
            "rank": 4,
            "dtype": "f32",
        },
        {
            "name": "Y",
            "rank": 4,
            "dtype": "f32",
        },
        {
            "name": "Out",
            "rank": 4,
            "dtype": "f32",
        },
    ],
    "shape": [1, 96, 18, 18],
    "constexprs": {
        "COUNT": 31104,
        "MULTIPLY": False,
        "BLOCK": 128,
    },
    "grid": [243, 1, 1],
    "roles": [],
    "metadata": {
        "dtype": "efficientnet-b3",
        "model": "timm/efficientnet_b3.ra2_in1k",
    },
}
