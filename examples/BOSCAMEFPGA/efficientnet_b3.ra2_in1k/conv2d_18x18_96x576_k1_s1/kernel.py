"""Standalone Triton source for conv2d_18x18_96x576_k1_s1."""

import triton
import triton.language as tl

@triton.jit
def conv2d_nchw(
    X, Weight, Bias, Out,
    H: tl.constexpr, W: tl.constexpr, C: tl.constexpr, OC: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BW: tl.constexpr,
):
    """1x1 stride-1 convolution. One program owns one output channel and a row tile."""
    tl.static_assert(KH == 1)
    tl.static_assert(KW == 1)
    tl.static_assert(SH == 1)
    tl.static_assert(SW == 1)
    tl.static_assert(PH == 0)
    tl.static_assert(PW == 0)
    tl.static_assert(BM > 0)
    tl.static_assert(BN > 0)
    tl.static_assert(BK > 0)
    oc = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2) * BW + tl.arange(0, BW)
    acc = tl.zeros((BW,), dtype=tl.float32)
    for ic in tl.range(C, loop_unroll_factor=1):
        x = tl.load(X + ic * H * W + oh * W + ow, ow < OW, other=0.0)
        w = tl.load(Weight + oc * C + ic)
        acc += x * w
    tl.store(
        Out + oc * OH * OW + oh * OW + ow,
        acc + tl.load(Bias + oc),
        ow < OW,
    )


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
        96,
        18,
        18
    ],
    "constexprs": {
        "H": 18,
        "W": 18,
        "C": 96,
        "OC": 576,
        "KH": 1,
        "KW": 1,
        "SH": 1,
        "SW": 1,
        "PH": 0,
        "PW": 0,
        "OH": 18,
        "OW": 18,
        "BM": 16,
        "BN": 16,
        "BK": 64,
        "BW": 16
    },
    "grid": [
        576,
        18,
        2
    ],
    "roles": [],
    "metadata": {
        "dtype": "efficientnet-b3",
        "model": "timm/efficientnet_b3.ra2_in1k"
    }
}
