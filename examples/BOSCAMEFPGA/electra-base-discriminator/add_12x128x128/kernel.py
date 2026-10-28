"""Standalone Triton source for add_12x128x128."""

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
 'arguments': [{'name': 'X', 'rank': 3, 'dtype': 'f32'},
               {'name': 'Y', 'rank': 3, 'dtype': 'f32'},
               {'name': 'Out', 'rank': 3, 'dtype': 'f32'}],
 'shape': [12, 128, 128],
 'constexprs': {'COUNT': 196608, 'BLOCK': 128},
 'grid': [1536, 1, 1],
 'roles': [],
 'metadata': {'dtype': 'f32',
              'model': 'google/electra-base-discriminator',
              'seq_len': 128,
              'note': 'Unique encoder shapes at seq_len 128. Twelve layers share these '
                      'bins.'}}
