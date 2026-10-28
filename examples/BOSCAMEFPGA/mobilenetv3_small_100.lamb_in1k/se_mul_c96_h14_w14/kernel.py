"""Standalone Triton source for se_mul_c96_h14_w14."""

import triton
import triton.language as tl

@triton.jit
def se_mul(X, Scale, Out, COUNT: tl.constexpr, SPATIAL: tl.constexpr):
    """One scalar program per element. Channel scale is a scalar load."""
    index = tl.program_id(0)
    channel = index // SPATIAL
    x = tl.load(X + index)
    scale = tl.load(Scale + channel)
    tl.store(Out + index, x * scale)

# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {'family': 'se_mul',
 'kernel': 'se_mul',
 'arguments': [{'name': 'X', 'rank': 1, 'dtype': 'f32'},
               {'name': 'Scale', 'rank': 1, 'dtype': 'f32'},
               {'name': 'Out', 'rank': 1, 'dtype': 'f32'}],
 'shape': [1, 96, 14, 14],
 'constexprs': {'COUNT': 18816, 'SPATIAL': 196},
 'grid': [18816, 1, 1],
 'roles': []}
