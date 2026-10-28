"""Standalone Triton source for pwconv_cin72_cout288_h1_w1."""

import triton
import triton.language as tl

@triton.jit
def pointwise_conv2d(X, Weight, Bias, Out, CIN: tl.constexpr, COUT: tl.constexpr,
                     H: tl.constexpr, W: tl.constexpr):
    """1x1 convolution. One program owns one output pixel."""
    oc = tl.program_id(0)
    spatial = tl.program_id(1)
    acc = 0.0
    for ic in range(CIN):
        x = tl.load(X + ic * H * W + spatial)
        wt = tl.load(Weight + oc * CIN + ic)
        acc += x * wt
    tl.store(Out + oc * H * W + spatial, acc + tl.load(Bias + oc))

# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {'family': 'pointwise_conv2d',
 'kernel': 'pointwise_conv2d',
 'arguments': [{'name': 'X', 'rank': 4, 'dtype': 'f32'},
               {'name': 'Weight', 'rank': 4, 'dtype': 'f32'},
               {'name': 'Bias', 'rank': 1, 'dtype': 'f32'},
               {'name': 'Out', 'rank': 4, 'dtype': 'f32'}],
 'shape': [1, 72, 1, 1],
 'constexprs': {'CIN': 72, 'COUT': 288, 'H': 1, 'W': 1},
 'grid': [288, 1, 1],
 'roles': []}
