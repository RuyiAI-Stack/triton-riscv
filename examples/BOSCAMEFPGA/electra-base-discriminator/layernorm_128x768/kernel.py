"""Standalone Triton source for layernorm_128x768."""

import triton
import triton.language as tl


@triton.jit
def layernorm(
    X,
    Weight,
    Bias,
    Out,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    """LayerNorm over the final (hidden) dimension, with affine parameters."""
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    valid = channel < WIDTH
    x = tl.load(X + row * WIDTH + channel, valid, other=0.0).to(tl.float32)
    weight = tl.load(Weight + channel, valid, other=1.0).to(tl.float32)
    bias = tl.load(Bias + channel, valid, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / WIDTH
    centered = tl.where(valid, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / WIDTH
    out = centered * tl.rsqrt(variance + EPS) * weight + bias
    tl.store(Out + row * WIDTH + channel, out, valid)

# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {'family': 'layernorm',
 'kernel': 'layernorm',
 'arguments': [{'name': 'X', 'rank': 2, 'dtype': 'f32'},
               {'name': 'Weight', 'rank': 1, 'dtype': 'f32'},
               {'name': 'Bias', 'rank': 1, 'dtype': 'f32'},
               {'name': 'Out', 'rank': 2, 'dtype': 'f32'}],
 'shape': [128, 768],
 'constexprs': {'ROWS': 128, 'WIDTH': 768, 'BLOCK': 1024, 'EPS': 1e-12},
 'grid': [128, 1, 1],
 'roles': [],
 'metadata': {'dtype': 'f32',
              'model': 'google/electra-base-discriminator',
              'seq_len': 128,
              'note': 'Unique encoder shapes at seq_len 128. Twelve layers share these '
                      'bins.'}}
