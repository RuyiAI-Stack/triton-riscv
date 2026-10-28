"""Standalone Triton source for dwconv_c144_h14_k5_s1_p2."""

import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d(X, Weight, Bias, Out, C: tl.constexpr, H: tl.constexpr,
                     W: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
                     SH: tl.constexpr, SW: tl.constexpr, PH: tl.constexpr,
                     PW: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr):
    """One program owns one output pixel so every load stays scalar."""
    ch = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)
    acc = 0.0
    for kh in range(KH):
        ih = oh * SH - PH + kh
        for kw in range(KW):
            iw = ow * SW - PW + kw
            if (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W):
                x = tl.load(X + ch * H * W + ih * W + iw)
                wt = tl.load(Weight + (ch * KH + kh) * KW + kw)
                acc += x * wt
    tl.store(Out + (ch * OH + oh) * OW + ow, acc + tl.load(Bias + ch))

# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {'family': 'depthwise_conv2d',
 'kernel': 'depthwise_conv2d',
 'arguments': [{'name': 'X', 'rank': 4, 'dtype': 'f32'},
               {'name': 'Weight', 'rank': 4, 'dtype': 'f32'},
               {'name': 'Bias', 'rank': 1, 'dtype': 'f32'},
               {'name': 'Out', 'rank': 4, 'dtype': 'f32'}],
 'shape': [1, 144, 14, 14],
 'constexprs': {'C': 144,
                'H': 14,
                'W': 14,
                'KH': 5,
                'KW': 5,
                'SH': 1,
                'SW': 1,
                'PH': 2,
                'PW': 2,
                'OH': 14,
                'OW': 14},
 'grid': [144, 14, 14],
 'roles': []}
