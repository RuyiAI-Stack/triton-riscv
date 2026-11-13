"""
Standalone Triton kernel for FPGA case 'bias_add_128x1536' (model
'e5-small-v2').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in
examples/BOSCAMEFPGA/e5-small-v2/bias_add_128x1536 before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def bias_add(
    X, Bias, Out, ROWS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    """Add a broadcast row bias without vector remainder/modulo operations."""
    row = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = channel < WIDTH
    index = row * WIDTH + channel
    x = tl.load(X + index, valid, other=0.0)
    bias = tl.load(Bias + channel, channel < WIDTH, other=0.0)
    tl.store(Out + index, x + bias, valid)


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "bias_add",
    "kernel": "bias_add",
    "arguments": [
        {"name": "X", "rank": 2, "dtype": "f32"},
        {"name": "Bias", "rank": 1, "dtype": "f32"},
        {"name": "Out", "rank": 2, "dtype": "f32"},
    ],
    "shape": [128, 1536],
    "argument_shapes": [[128, 1536], [1536], [128, 1536]],
    "constexprs": {"ROWS": 128, "WIDTH": 1536, "BLOCK": 2048},
    "grid": [128, 1, 1],
    "roles": [],
    "outputs": [2],
    "checked_elements": 196608,
    "metadata": {
        "model": "intfloat/e5-small-v2",
        "batch": 1,
        "sequence_length": 128,
        "target": "nr-fpga",
    },
}
