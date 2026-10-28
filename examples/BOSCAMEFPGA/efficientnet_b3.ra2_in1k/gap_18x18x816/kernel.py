"""Standalone Triton source for gap_18x18x816."""

import triton
import triton.language as tl

@triton.jit
def gap(X, Out, SPATIAL: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    """Global average pool. One program reduces one channel along unit-stride pixels."""
    tl.static_assert(C > 0)
    ch = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for block in range(tl.cdiv(SPATIAL, BLOCK)):
        pixel = block * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(X + ch * SPATIAL + pixel, pixel < SPATIAL, other=0.0)
    tl.store(Out + ch, tl.div_rn(tl.sum(acc), SPATIAL * 1.0))


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "gap",
    "kernel": "gap",
    "arguments": [
        {
            "name": "X",
            "rank": 4,
            "dtype": "f32"
        },
        {
            "name": "Out",
            "rank": 2,
            "dtype": "f32"
        }
    ],
    "shape": [
        1,
        816,
        18,
        18
    ],
    "constexprs": {
        "SPATIAL": 324,
        "C": 816,
        "BLOCK": 128
    },
    "grid": [
        816,
        1,
        1
    ],
    "roles": [],
    "metadata": {
        "dtype": "efficientnet-b3",
        "model": "timm/efficientnet_b3.ra2_in1k"
    }
}
