"""Standalone Triton source for bias_add_128x1."""

import triton
import triton.language as tl


@triton.jit
def bias_add(
    X, Bias, Out, ROWS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    """Add a broadcast row bias without vector remainder/modulo operations."""
    row = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (row < ROWS) & (channel < WIDTH)
    index = row * WIDTH + channel
    x = tl.load(X + index, valid, other=0.0)
    bias = tl.load(Bias + channel, channel < WIDTH, other=0.0)
    tl.store(Out + index, x + bias, valid)

# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {'family': 'bias_add',
 'kernel': 'bias_add',
 'arguments': [{'name': 'X', 'rank': 2, 'dtype': 'f32'},
               {'name': 'Bias', 'rank': 1, 'dtype': 'f32'},
               {'name': 'Out', 'rank': 2, 'dtype': 'f32'}],
 'shape': [128, 1],
 'constexprs': {'ROWS': 128, 'WIDTH': 1, 'BLOCK': 1024},
 'grid': [128, 1, 1],
 'roles': [],
 'metadata': {'dtype': 'f32',
              'model': 'google/electra-base-discriminator',
              'seq_len': 128,
              'note': 'Unique encoder shapes at seq_len 128. Twelve layers share these '
                      'bins.'}}
