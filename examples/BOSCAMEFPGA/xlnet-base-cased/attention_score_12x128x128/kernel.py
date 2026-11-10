"""Standalone Triton kernel for FPGA case 'attention_score_12x128x128' (model 'xlnet-base-cased').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/xlnet-base-cased/attention_score_12x128x128
before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def attention_score(
    Content,
    Position,
    Segment,
    Mask,
    Out,
    SEQ: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    h = tl.program_id(0)
    row = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    offset = (h * SEQ + row) * SEQ + col
    ac = tl.load(Content + offset)
    bd = tl.load(Position + offset)
    ef = tl.load(Segment + offset)
    # Host creates XLNet content-stream non-target mask, diagonal explicitly zero.
    blocked = tl.load(Mask + row * SEQ + col).to(tl.float32)
    tl.store(Out + offset, ((ac + bd) + ef) * SCALE - blocked * 1.0e30)


KERNEL_CONFIG = {
    "family": "attention_score",
    "kernel": "attention_score",
    "arguments": [
        {"name": "Content", "rank": 3, "dtype": "f32"},
        {"name": "Position", "rank": 3, "dtype": "f32"},
        {"name": "Segment", "rank": 3, "dtype": "f32"},
        {"name": "Mask", "rank": 2, "dtype": "i32"},
        {"name": "Out", "rank": 3, "dtype": "f32"},
    ],
    "argument_shapes": [
        [12, 128, 128],
        [12, 128, 128],
        [12, 128, 128],
        [128, 128],
        [12, 128, 128],
    ],
    "shape": [12, 128, 128],
    "constexprs": {"SEQ": 128, "SCALE": 0.125, "BLOCK": 128},
    "grid": [12, 128, 1],
    "roles": ["xlnet_model"],
    "metadata": {"model": "xlnet/xlnet-base-cased"},
}
