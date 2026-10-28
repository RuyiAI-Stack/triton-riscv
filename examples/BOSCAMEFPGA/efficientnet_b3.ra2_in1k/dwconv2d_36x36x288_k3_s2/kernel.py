"""Standalone Triton source for dwconv2d_36x36x288_k3_s2."""

import triton
import triton.language as tl

@triton.jit
def dwconv2d_nchw(
    X, Weight, Bias, Out,
    H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Depthwise NCHW convolution. One program owns one output pixel."""
    tl.static_assert(BLOCK > 0)
    tl.static_assert(C > 0)
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
KERNEL_CONFIG = {
    "family": "dwconv2d_f32",
    "kernel": "dwconv2d_nchw",
    "arguments": [
        {
            "name": "X",
            "rank": 4,
            "dtype": "f32"
        },
        {
            "name": "Weight",
            "rank": 4,
            "dtype": "f32"
        },
        {
            "name": "Bias",
            "rank": 1,
            "dtype": "f32"
        },
        {
            "name": "Out",
            "rank": 4,
            "dtype": "f32"
        }
    ],
    "shape": [
        1,
        288,
        36,
        36
    ],
    "constexprs": {
        "H": 36,
        "W": 36,
        "C": 288,
        "KH": 3,
        "KW": 3,
        "SH": 2,
        "SW": 2,
        "PH": 1,
        "PW": 1,
        "OH": 18,
        "OW": 18,
        "BLOCK": 16
    },
    "grid": [
        288,
        18,
        18
    ],
    "roles": [],
    "metadata": {
        "dtype": "efficientnet-b3",
        "model": "timm/efficientnet_b3.ra2_in1k"
    }
}
