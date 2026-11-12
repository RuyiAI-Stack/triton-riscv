"""Standalone Triton kernel for FPGA case 'split_heads_12x128x64' (model 'xlnet-base-cased').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/xlnet-base-cased/split_heads_12x128x64
before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def split_heads(
    X,
    Out,
    LENGTH: tl.constexpr,
    WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    h = tl.program_id(0)
    row = tl.program_id(1)
    c = tl.arange(0, BLOCK)
    value = tl.load(X + row * WIDTH + h * HEAD_DIM + c)
    tl.store(Out + (h * LENGTH + row) * HEAD_DIM + c, value)


KERNEL_CONFIG = {
    "family": "split_heads",
    "kernel": "split_heads",
    "arguments": [
        {"name": "X", "rank": 2, "dtype": "f32"},
        {"name": "Out", "rank": 3, "dtype": "f32"},
    ],
    "argument_shapes": [[128, 768], [12, 128, 64]],
    "shape": [12, 128, 64],
    "constexprs": {"LENGTH": 128, "WIDTH": 768, "HEAD_DIM": 64, "BLOCK": 64},
    "grid": [12, 128, 1],
    "roles": ["xlnet_model"],
    "metadata": {"model": "xlnet/xlnet-base-cased"},
}
