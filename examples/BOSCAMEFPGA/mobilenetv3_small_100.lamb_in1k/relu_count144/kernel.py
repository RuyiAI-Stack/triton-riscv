"""Standalone Triton source for relu_count144."""

import triton
import triton.language as tl

@triton.jit
def relu(X, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    x = tl.load(X + index, valid, other=0.0)
    tl.store(Out + index, tl.maximum(x, 0.0), valid)

# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {'family': 'relu',
 'kernel': 'relu',
 'arguments': [{'name': 'X', 'rank': 1, 'dtype': 'f32'},
               {'name': 'Out', 'rank': 1, 'dtype': 'f32'}],
 'shape': [144],
 'constexprs': {'COUNT': 144, 'BLOCK': 128},
 'grid': [2, 1, 1],
 'roles': []}
