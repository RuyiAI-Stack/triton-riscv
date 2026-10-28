"""Standalone Triton source for hardsigmoid_count120."""

import triton
import triton.language as tl

@triton.jit
def hardsigmoid(X, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    x = tl.load(X + index, valid, other=0.0)
    shifted = x + 3.0
    lower = tl.maximum(shifted, 0.0)
    clamped = tl.minimum(lower, 6.0)
    # Round the FP32 division, rather than multiplying by a rounded reciprocal.
    tl.store(Out + index, tl.div_rn(clamped, 6.0), valid)

# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {'family': 'hardsigmoid',
 'kernel': 'hardsigmoid',
 'arguments': [{'name': 'X', 'rank': 1, 'dtype': 'f32'},
               {'name': 'Out', 'rank': 1, 'dtype': 'f32'}],
 'shape': [120],
 'constexprs': {'COUNT': 120, 'BLOCK': 128},
 'grid': [1, 1, 1],
 'roles': []}
