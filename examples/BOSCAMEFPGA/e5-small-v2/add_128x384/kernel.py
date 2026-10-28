"""Standalone Triton source for add_128x384."""

import triton
import triton.language as tl


@triton.jit
def add(X, Y, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    tl.store(
        Out + index,
        tl.load(X + index, valid, other=0.0)
        + tl.load(Y + index, valid, other=0.0),
        valid,
    )

# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {'family': 'add',
 'kernel': 'add',
 'arguments': [{'name': 'X', 'rank': 2, 'dtype': 'f32'},
               {'name': 'Y', 'rank': 2, 'dtype': 'f32'},
               {'name': 'Out', 'rank': 2, 'dtype': 'f32'}],
 'shape': [128, 384],
 'constexprs': {'COUNT': 49152, 'BLOCK': 128},
 'grid': [384, 1, 1],
 'roles': [],
 'metadata': {'dtype': 'f32',
              'model': 'intfloat/e5-small-v2',
              'seq_len': 128,
              'note': 'Unique encoder shapes at seq_len 128. Twelve layers share these '
                      'bins.'}}
