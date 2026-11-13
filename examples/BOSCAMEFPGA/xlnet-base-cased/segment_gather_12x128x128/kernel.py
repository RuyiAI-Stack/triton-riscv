"""Standalone Triton kernel for FPGA case 'segment_gather_12x128x128' (model 'xlnet-base-cased').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/xlnet-base-cased/segment_gather_12x128x128
before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def segment_gather(X, Segments, Out, SEQ: tl.constexpr, BLOCK: tl.constexpr):
    h = tl.program_id(0)
    row = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    # Host-generated one-hot segment relation, identical to HF seg_mat.
    same_weight = tl.load(Segments + row * SEQ + col)
    different_weight = tl.load(Segments + SEQ * SEQ + row * SEQ + col)
    same = tl.load(X + (h * SEQ + row) * 2)
    different = tl.load(X + (h * SEQ + row) * 2 + 1)
    value = same * same_weight + different * different_weight
    tl.store(Out + (h * SEQ + row) * SEQ + col, value)


KERNEL_CONFIG = {
    "family": "segment_gather",
    "kernel": "segment_gather",
    "arguments": [
        {"name": "X", "rank": 3, "dtype": "f32"},
        {"name": "Segments", "rank": 3, "dtype": "f32"},
        {"name": "Out", "rank": 3, "dtype": "f32"},
    ],
    "argument_shapes": [[12, 128, 2], [2, 128, 128], [12, 128, 128]],
    "shape": [12, 128, 128],
    "constexprs": {"SEQ": 128, "BLOCK": 128},
    "grid": [12, 128, 1],
    "roles": ["xlnet_model"],
    "metadata": {"model": "xlnet/xlnet-base-cased"},
}
