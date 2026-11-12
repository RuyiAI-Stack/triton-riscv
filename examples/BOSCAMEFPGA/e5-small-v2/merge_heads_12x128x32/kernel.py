"""Standalone Triton kernel for FPGA case 'merge_heads_12x128x32' (model 'e5-small-v2').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/e5-small-v2/merge_heads_12x128x32
before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def head_layout(
    X,
    Out,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    head = tl.program_id(0)
    row = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    valid = col < DIM
    flat = row * (HEADS * DIM) + head * DIM + col
    packed = (head * SEQ + row) * DIM + col
    if SPLIT:
        value = tl.load(X + flat, valid, other=0.0)
        tl.store(Out + packed, value, valid)
    else:
        value = tl.load(X + packed, valid, other=0.0)
        tl.store(Out + flat, value, valid)


KERNEL_CONFIG = {
    "family": "head_layout",
    "kernel": "head_layout",
    "arguments": [
        {"name": "X", "rank": 3, "dtype": "f32"},
        {"name": "Out", "rank": 2, "dtype": "f32"},
    ],
    "shape": [12, 128, 32],
    "argument_shapes": [[12, 128, 32], [128, 384]],
    "constexprs": {"SEQ": 128, "HEADS": 12, "DIM": 32, "SPLIT": False, "BLOCK": 32},
    "grid": [12, 128, 1],
    "roles": [],
    "outputs": [1],
    "checked_elements": 49152,
    "metadata": {
        "model": "intfloat/e5-small-v2",
        "batch": 1,
        "sequence_length": 128,
        "target": "nr-fpga",
    },
}
