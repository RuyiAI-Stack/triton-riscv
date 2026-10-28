"""Standalone Triton source for add_count18816."""

import triton
import triton.language as tl

@triton.jit
def residual_add(X, Y, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    x = tl.load(X + index, valid, other=0.0)
    y = tl.load(Y + index, valid, other=0.0)
    tl.store(Out + index, x + y, valid)

# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {'family': 'residual_add',
 'kernel': 'residual_add',
 'arguments': [{'name': 'X', 'rank': 1, 'dtype': 'f32'},
               {'name': 'Y', 'rank': 1, 'dtype': 'f32'},
               {'name': 'Out', 'rank': 1, 'dtype': 'f32'}],
 'shape': [18816],
 'constexprs': {'COUNT': 18816, 'BLOCK': 128},
 'grid': [147, 1, 1],
 'roles': []}
