"""Standalone Triton source for pwconv_cin88_cout24_h28_w28."""

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
 'shape': [1, 88, 28, 28],
 'constexprs': {'CIN': 88, 'COUT': 24, 'H': 28, 'W': 28},
 'grid': [24, 784, 1],
 'roles': []}
