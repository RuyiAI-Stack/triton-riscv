"""Standalone Triton kernel for FPGA case 'relative_shift_12x128x128' (model 'xlnet-base-cased').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/xlnet-base-cased/relative_shift_12x128x128
before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def relative_shift(X, Out, SEQ: tl.constexpr, REL: tl.constexpr, BLOCK: tl.constexpr):
    h = tl.program_id(0)
    row = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    # HF rel_shift_bnij: reshape(H,REL,SEQ), drop [:,1:,:], reshape(H,SEQ,REL-1).
    value = tl.load(
        X + h * SEQ * REL + SEQ + row * (REL - 1) + col, col < SEQ, other=0.0
    )
    tl.store(Out + (h * SEQ + row) * SEQ + col, value, col < SEQ)


KERNEL_CONFIG = {
    "family": "relative_shift",
    "kernel": "relative_shift",
    "arguments": [
        {"name": "X", "rank": 3, "dtype": "f32"},
        {"name": "Out", "rank": 3, "dtype": "f32"},
    ],
    "argument_shapes": [[12, 128, 256], [12, 128, 128]],
    "shape": [12, 128, 128],
    "constexprs": {"SEQ": 128, "REL": 256, "BLOCK": 128},
    "grid": [12, 128, 1],
    "roles": ["xlnet_model"],
    "metadata": {"model": "xlnet/xlnet-base-cased"},
}
