"""Standalone Triton source for gelu_128x1536."""

import triton
import triton.language as tl


@triton.jit
def gelu(X, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    """Exact Hugging Face GELU (approximate=none, erf form)."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    x = tl.load(X + index, valid, other=0.0).to(tl.float32)
    y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
    tl.store(Out + index, y, valid)

# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {'family': 'gelu',
 'kernel': 'gelu',
 'arguments': [{'name': 'X', 'rank': 2, 'dtype': 'f32'},
               {'name': 'Out', 'rank': 2, 'dtype': 'f32'}],
 'shape': [128, 1536],
 'constexprs': {'COUNT': 196608, 'BLOCK': 128},
 'grid': [1536, 1, 1],
 'roles': [],
 'math_accuracy': {'erf': {'domain': 'all-f32',
                           'special_values': 'ieee',
                           'max_abs_error': 2e-06,
                           'max_rel_error': 0.0}},
 'metadata': {'dtype': 'f32',
              'model': 'thenlper/gte-small',
              'seq_len': 128,
              'note': 'Unique encoder shapes at seq_len 128. Twelve layers share these '
                      'bins.'}}
