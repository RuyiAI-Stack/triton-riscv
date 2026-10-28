"""Standalone Triton source for conv2d_288x288_3x40_k3_s2."""

import triton
import triton.language as tl

@triton.jit
def conv2d_nchw(
    X, Weight, Bias, Out,
    H: tl.constexpr, W: tl.constexpr, C: tl.constexpr, OC: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """NCHW convolution. One program owns one output pixel so every load is scalar."""
    tl.static_assert(BM > 0)
    tl.static_assert(BN > 0)
    tl.static_assert(BK > 0)
    oc = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)
    acc = 0.0
    for ic in range(C):
        for kh in range(KH):
            ih = oh * SH - PH + kh
            for kw in range(KW):
                iw = ow * SW - PW + kw
                if (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W):
                    x = tl.load(X + ic * H * W + ih * W + iw)
                    wt = tl.load(Weight + ((oc * C + ic) * KH + kh) * KW + kw)
                    acc += x * wt
    tl.store(Out + (oc * OH + oh) * OW + ow, acc + tl.load(Bias + oc))


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "conv2d_f32",
    "kernel": "conv2d_nchw",
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
        3,
        288,
        288
    ],
    "constexprs": {
        "H": 288,
        "W": 288,
        "C": 3,
        "OC": 40,
        "KH": 3,
        "KW": 3,
        "SH": 2,
        "SW": 2,
        "PH": 1,
        "PW": 1,
        "OH": 144,
        "OW": 144,
        "BM": 16,
        "BN": 16,
        "BK": 64
    },
    "grid": [
        40,
        144,
        144
    ],
    "roles": [],
    "metadata": {
        "dtype": "efficientnet-b3",
        "model": "timm/efficientnet_b3.ra2_in1k"
    }
}
