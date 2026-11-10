"""Standalone Triton kernel for FPGA case 'masked_softmax_12x128x128' (model 'electra-base-discriminator').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/electra-base-discriminator/masked_softmax_12x128x128
before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def masked_softmax(X, Mask, Out, SEQ: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    keep = tl.load(Mask + col, col < SEQ, other=0) != 0
    x = tl.load(X + row * SEQ + col, col < SEQ, other=0.0)
    x = tl.where(keep, x, -float("inf"))
    e = tl.exp(x - tl.max(x, axis=0))
    y = e / tl.sum(e, axis=0)
    tl.store(Out + row * SEQ + col, y, col < SEQ)


KERNEL_CONFIG = {
    "family": "masked_softmax",
    "kernel": "masked_softmax",
    "arguments": [
        {"name": "X", "rank": 3, "dtype": "f32"},
        {"name": "Mask", "rank": 1, "dtype": "i32"},
        {"name": "Out", "rank": 3, "dtype": "f32"},
    ],
    "argument_shapes": [[12, 128, 128], [128], [12, 128, 128]],
    "shape": [12, 128, 128],
    "constexprs": {"SEQ": 128, "BLOCK": 128},
    "grid": [1536, 1, 1],
    "roles": ["model"],
    "metadata": {"model": "google/electra-base-discriminator"},
    "math_accuracy": {
        "exp": {
            "domain": "all-f32",
            "special_values": "ieee",
            "max_abs_error": 1.1210387714598537e-44,
            "max_rel_error": 2e-06,
        }
    },
}
