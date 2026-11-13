"""
Standalone Triton kernel for FPGA case 'softmax_12x128x128' (model
'e5-small-v2').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in
examples/BOSCAMEFPGA/e5-small-v2/softmax_12x128x128 before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def softmax(X, Out, ROWS: tl.constexpr, LENGTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    valid = column < LENGTH
    x = tl.load(X + row * LENGTH + column, valid, other=-float("inf"))
    maximum = tl.max(x, axis=0)
    exponential = tl.exp(x - maximum)
    total = tl.sum(exponential, axis=0)
    tl.store(Out + row * LENGTH + column, exponential / total, valid)


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "softmax",
    "kernel": "softmax",
    "arguments": [
        {"name": "X", "rank": 3, "dtype": "f32"},
        {"name": "Out", "rank": 3, "dtype": "f32"},
    ],
    "shape": [12, 128, 128],
    "argument_shapes": [[12, 128, 128], [12, 128, 128]],
    "constexprs": {"ROWS": 1536, "LENGTH": 128, "BLOCK": 128},
    "grid": [1536, 1, 1],
    "roles": [],
    "math_accuracy": {
        "exp": {
            "domain": "all-f32",
            "special_values": "ieee",
            "max_abs_error": 1.1210387714598537e-44,
            "max_rel_error": 2e-06,
        }
    },
    "outputs": [1],
    "checked_elements": 196608,
    "metadata": {
        "model": "intfloat/e5-small-v2",
        "batch": 1,
        "sequence_length": 128,
        "target": "nr-fpga",
    },
}
