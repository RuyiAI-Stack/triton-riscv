"""Standalone Triton source for quantize_1x1536."""

import triton
import triton.language as tl

@triton.jit
def quantize(X, Q, Scale, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    """Per-row signed INT8 quantization; half rounds away from zero."""
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    valid = channel < WIDTH
    x = tl.load(X + row * WIDTH + channel, valid, other=0.0).to(tl.float32)
    maximum = tl.max(tl.abs(x), axis=0)
    scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
    value = tl.minimum(tl.maximum(tl.div_rn(x, scale), -127.0), 127.0)
    # Same-width f32->i32, then a narrowing integer store. A direct f32->i8
    # cast lowers to vfncvt.rtz.x.f.w, which the FPGA assembler rejects.
    base = value.to(tl.int32)
    fraction = value - base.to(tl.float32)
    adjustment = tl.where(
        tl.abs(fraction) >= 0.5, tl.where(value >= 0, 1, -1), 0
    )
    integer = (base + adjustment).to(tl.int8)
    tl.store(Scale + row, scale)
    tl.store(Q + row * WIDTH + channel, integer, valid)


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "per_token_quantization",
    "kernel": "quantize",
    "arguments": [
        {
            "name": "X",
            "rank": 2,
            "dtype": "f32",
        },
        {
            "name": "Q",
            "rank": 2,
            "dtype": "i8",
        },
        {
            "name": "Scale",
            "rank": 1,
            "dtype": "f32",
        },
    ],
    "shape": [1, 1536],
    "constexprs": {
        "WIDTH": 1536,
        "BLOCK": 2048,
    },
    "grid": [1, 1, 1],
    "roles": [],
    "metadata": {
        "dtype": "efficientnet-b3",
        "model": "timm/efficientnet_b3.ra2_in1k",
    },
}
