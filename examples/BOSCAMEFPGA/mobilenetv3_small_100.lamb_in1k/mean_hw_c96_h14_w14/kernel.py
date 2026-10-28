"""Standalone Triton source for mean_hw_c96_h14_w14."""

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
KERNEL_CONFIG = {'family': 'mean_hw',
 'kernel': 'mean_hw',
 'arguments': [{'name': 'X', 'rank': 1, 'dtype': 'f32'},
               {'name': 'Out', 'rank': 1, 'dtype': 'f32'}],
 'shape': [1, 96, 14, 14],
 'constexprs': {'SPATIAL': 196, 'C': 96, 'BLOCK': 128},
 'grid': [96, 1, 1],
 'roles': []}
