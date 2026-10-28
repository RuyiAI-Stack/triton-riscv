"""Standalone Triton source for conv_stem_cin3_cout16_h224_k3_s2_p1."""

import triton
import triton.language as tl

@triton.jit
def conv_stem(X, Weight, Bias, Out, CIN: tl.constexpr, COUT: tl.constexpr,
              H: tl.constexpr, W: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
              SH: tl.constexpr, SW: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
              OH: tl.constexpr, OW: tl.constexpr):
    """One program owns one output pixel so the 3x3 gather stays scalar."""
    oc = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)
    acc = 0.0
    for ic in range(CIN):
        for kh in range(KH):
            ih = oh * SH - PH + kh
            for kw in range(KW):
                iw = ow * SW - PW + kw
                if (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W):
                    x = tl.load(X + (ic * H + ih) * W + iw)
                    wt = tl.load(Weight + ((oc * CIN + ic) * KH + kh) * KW + kw)
                    acc += x * wt
    tl.store(Out + (oc * OH + oh) * OW + ow, acc + tl.load(Bias + oc))

# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {'family': 'conv_stem',
 'kernel': 'conv_stem',
 'arguments': [{'name': 'X', 'rank': 4, 'dtype': 'f32'},
               {'name': 'Weight', 'rank': 4, 'dtype': 'f32'},
               {'name': 'Bias', 'rank': 1, 'dtype': 'f32'},
               {'name': 'Out', 'rank': 4, 'dtype': 'f32'}],
 'shape': [1, 3, 224, 224],
 'constexprs': {'CIN': 3,
                'COUT': 16,
                'H': 224,
                'W': 224,
                'KH': 3,
                'KW': 3,
                'SH': 2,
                'SW': 2,
                'PH': 1,
                'PW': 1,
                'OH': 112,
                'OW': 112},
 'grid': [16, 112, 112],
 'roles': []}
