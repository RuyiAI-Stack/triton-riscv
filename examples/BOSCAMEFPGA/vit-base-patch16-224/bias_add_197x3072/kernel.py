"""
Standalone Triton kernel for FPGA case 'bias_add_197x3072' (model
'vit-base-patch16-224').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in
examples/BOSCAMEFPGA/vit-base-patch16-224/bias_add_197x3072 before linking
launch.c.
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
    valid = (row < ROWS) & (channel < WIDTH)
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
    "shape": [197, 3072],
    "constexprs": {"ROWS": 197, "WIDTH": 3072, "BLOCK": 1024},
    "grid": [197, 3, 1],
    "roles": [],
    "metadata": {"dtype": "f32", "model": "google/vit-base-patch16-224"},
}
