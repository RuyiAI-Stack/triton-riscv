"""Standalone Triton kernel for FPGA case 'unpack_heads_12x128x64' (model 'electra-base-discriminator').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/electra-base-discriminator/unpack_heads_12x128x64
before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def unpack_heads(
    X,
    Out,
    HEADS: tl.constexpr,
    SEQ: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + head * SEQ * DIM + row * DIM + col, col < DIM, other=0.0)
    tl.store(Out + row * HEADS * DIM + head * DIM + col, x, col < DIM)


KERNEL_CONFIG = {
    "family": "unpack_heads",
    "kernel": "unpack_heads",
    "arguments": [
        {"name": "X", "rank": 3, "dtype": "f32"},
        {"name": "Out", "rank": 2, "dtype": "f32"},
    ],
    "argument_shapes": [[12, 128, 64], [128, 768]],
    "shape": [12, 128, 64],
    "constexprs": {"HEADS": 12, "SEQ": 128, "DIM": 64, "BLOCK": 64},
    "grid": [128, 12, 1],
    "roles": ["model"],
    "metadata": {"model": "google/electra-base-discriminator"},
}
