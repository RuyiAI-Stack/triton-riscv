"""Standalone Triton source for pwconv_cin144_cout48_h14_w14."""

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
 'shape': [1, 144, 14, 14],
 'constexprs': {'CIN': 144, 'COUT': 48, 'H': 14, 'W': 14},
 'grid': [48, 196, 1],
 'roles': []}
