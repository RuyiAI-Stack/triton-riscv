"""Standalone Triton kernel for FPGA case 'add_128x384' (model 'e5-small-v2').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/e5-small-v2/add_128x384
before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def add(X, Y, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    tl.store(
        Out + index,
        tl.load(X + index, valid, other=0.0) + tl.load(Y + index, valid, other=0.0),
        valid,
    )


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "add",
    "kernel": "add",
    "arguments": [
        {"name": "X", "rank": 2, "dtype": "f32"},
        {"name": "Y", "rank": 2, "dtype": "f32"},
        {"name": "Out", "rank": 2, "dtype": "f32"},
    ],
    "shape": [128, 384],
    "argument_shapes": [[128, 384], [128, 384], [128, 384]],
    "constexprs": {"COUNT": 49152, "BLOCK": 128},
    "grid": [384, 1, 1],
    "roles": [],
    "outputs": [2],
    "checked_elements": 49152,
    "metadata": {
        "model": "intfloat/e5-small-v2",
        "batch": 1,
        "sequence_length": 128,
        "target": "nr-fpga",
    },
}
