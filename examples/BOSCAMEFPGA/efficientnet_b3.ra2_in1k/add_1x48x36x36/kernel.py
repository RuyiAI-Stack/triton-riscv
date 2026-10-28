"""Standalone Triton source for add_1x48x36x36."""

import triton
import triton.language as tl

@triton.jit
def binary(
    X, Y, Out, COUNT: tl.constexpr, MULTIPLY: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    x = tl.load(X + index, valid, other=0.0)
    y = tl.load(Y + index, valid, other=0.0)
    result = x * y if MULTIPLY else x + y
    tl.store(Out + index, result, valid)


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "add",
    "kernel": "binary",
    "arguments": [
        {
            "name": "X",
            "rank": 4,
            "dtype": "f32",
        },
        {
            "name": "Y",
            "rank": 4,
            "dtype": "f32",
        },
        {
            "name": "Out",
            "rank": 4,
            "dtype": "f32",
        },
    ],
    "shape": [1, 48, 36, 36],
    "constexprs": {
        "COUNT": 62208,
        "MULTIPLY": False,
        "BLOCK": 128,
    },
    "grid": [486, 1, 1],
    "roles": [],
    "metadata": {
        "dtype": "efficientnet-b3",
        "model": "timm/efficientnet_b3.ra2_in1k",
    },
}
