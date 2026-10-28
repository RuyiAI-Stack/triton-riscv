"""Standalone Triton source for dequantize_1x1000."""

import triton
import triton.language as tl

@triton.jit
def dequantize(
    X, Row, Column, Out,
    ROWS: tl.constexpr, COLS: tl.constexpr, BLOCK: tl.constexpr,
):
    """Out = int32 value * row scale * column scale. This case has one row."""
    tl.static_assert(ROWS == 1)
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COLS
    value = tl.load(X + index, valid, other=0).to(tl.float32)
    row = tl.load(Row)
    column = tl.load(Column + index, valid, other=0.0)
    tl.store(Out + index, value * row * column, valid)


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "dequantization",
    "kernel": "dequantize",
    "arguments": [
        {
            "name": "X",
            "rank": 2,
            "dtype": "i32"
        },
        {
            "name": "Row",
            "rank": 1,
            "dtype": "f32"
        },
        {
            "name": "Column",
            "rank": 1,
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
        1000
    ],
    "constexprs": {
        "ROWS": 1,
        "COLS": 1000,
        "BLOCK": 128
    },
    "grid": [
        8,
        1,
        1
    ],
    "roles": [],
    "metadata": {
        "dtype": "efficientnet-b3",
        "model": "timm/efficientnet_b3.ra2_in1k"
    }
}
