"""Standalone Triton kernel for FPGA case 'attention_mask_12x128x128' (model 'e5-small-v2').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/e5-small-v2/attention_mask_12x128x128
before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def attention_mask(
    X, Mask, Out, ROWS: tl.constexpr, LENGTH: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < LENGTH
    x = tl.load(X + row * LENGTH + col, valid, other=0.0)
    keep = tl.load(Mask + col, valid, other=0).to(tl.float32)
    # BERT uses finfo(float32).min as the additive mask. A padding query
    # remains a query; only key columns are masked.
    out = x + (keep - 1.0) * (3.4028234663852886e38)
    tl.store(Out + row * LENGTH + col, out, valid)


KERNEL_CONFIG = {
    "family": "attention_mask",
    "kernel": "attention_mask",
    "arguments": [
        {"name": "X", "rank": 3, "dtype": "f32"},
        {"name": "Mask", "rank": 1, "dtype": "i32"},
        {"name": "Out", "rank": 3, "dtype": "f32"},
    ],
    "shape": [12, 128, 128],
    "argument_shapes": [[12, 128, 128], [128], [12, 128, 128]],
    "constexprs": {"ROWS": 1536, "LENGTH": 128, "BLOCK": 128},
    "grid": [1536, 1, 1],
    "roles": ["bidirectional_key_padding_mask"],
    "outputs": [2],
    "checked_elements": 196608,
    "metadata": {
        "model": "intfloat/e5-small-v2",
        "batch": 1,
        "sequence_length": 128,
        "target": "nr-fpga",
    },
}
