"""
Standalone Triton kernel for FPGA case 'add_12x128x128' (model 'e5-small-v2').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/e5-small-v2/add_12x128x128
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
        {"name": "X", "rank": 3, "dtype": "f32"},
        {"name": "Y", "rank": 3, "dtype": "f32"},
        {"name": "Out", "rank": 3, "dtype": "f32"},
    ],
    "shape": [12, 128, 128],
    "constexprs": {"COUNT": 196608, "BLOCK": 128},
    "grid": [1536, 1, 1],
    "roles": [],
    "metadata": {
        "dtype": "f32",
        "model": "intfloat/e5-small-v2",
        "seq_len": 128,
        "note": "Unique encoder shapes at seq_len 128. Twelve layers share these bins.",
    },
}
