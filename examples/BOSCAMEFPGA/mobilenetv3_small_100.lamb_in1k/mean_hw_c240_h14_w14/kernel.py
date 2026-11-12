"""
Standalone Triton kernel for FPGA case 'mean_hw_c240_h14_w14' (model
'mobilenetv3_small_100.lamb_in1k').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in
examples/BOSCAMEFPGA/mobilenetv3_small_100.lamb_in1k/mean_hw_c240_h14_w14 before
linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def mean_hw(X, Out, SPATIAL: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    """One program reduces one channel, matching the EfficientNet GAP kernel."""
    ch = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for block in range(tl.cdiv(SPATIAL, BLOCK)):
        pixel = block * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(X + ch * SPATIAL + pixel, pixel < SPATIAL, other=0.0)
    tl.store(Out + ch, tl.div_rn(tl.sum(acc), SPATIAL * 1.0))


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "mean_hw",
    "kernel": "mean_hw",
    "arguments": [
        {"name": "X", "rank": 1, "dtype": "f32"},
        {"name": "Out", "rank": 1, "dtype": "f32"},
    ],
    "shape": [1, 240, 14, 14],
    "constexprs": {"SPATIAL": 196, "C": 240, "BLOCK": 128},
    "grid": [240, 1, 1],
    "roles": [],
}
