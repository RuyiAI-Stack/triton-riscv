"""Standalone Triton kernel for FPGA case 'merge_heads_128x768' (model 'xlnet-base-cased').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/xlnet-base-cased/merge_heads_128x768
before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def merge_heads(
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
    value = tl.load(X + (h * LENGTH + row) * HEAD_DIM + c)
    tl.store(Out + row * WIDTH + h * HEAD_DIM + c, value)


KERNEL_CONFIG = {
    "family": "merge_heads",
    "kernel": "merge_heads",
    "arguments": [
        {"name": "X", "rank": 3, "dtype": "f32"},
        {"name": "Out", "rank": 2, "dtype": "f32"},
    ],
    "argument_shapes": [[12, 128, 64], [128, 768]],
    "shape": [128, 768],
    "constexprs": {"LENGTH": 128, "WIDTH": 768, "HEAD_DIM": 64, "BLOCK": 64},
    "grid": [12, 128, 1],
    "roles": ["xlnet_model"],
    "metadata": {"model": "xlnet/xlnet-base-cased"},
}
