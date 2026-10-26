"""Standalone Triton source for quantize_1x768."""

import triton
import triton.language as tl


@triton.jit
def quantize(X, Q, Scale, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    """Per-row symmetric signed INT8 quantization, ties away from zero."""
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    valid = channel < WIDTH
    x = tl.load(X + row * WIDTH + channel, valid, other=0.0).to(tl.float32)
    maximum = tl.max(tl.abs(x), axis=0)
    scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
    value = tl.minimum(tl.maximum(tl.div_rn(x, scale), -127.0), 127.0)
    base = value.to(tl.int32)
    fraction = value - base.to(tl.float32)
    # Adding 0.5 in f32 first can round nextafter(0.5, 0) up to 1.0.
    adjustment = tl.where(
        tl.abs(fraction) >= 0.5, tl.where(value >= 0, 1, -1), 0
    )
    integer = (base + adjustment).to(tl.int8)
    tl.store(Scale + row, scale)
    tl.store(Q + row * WIDTH + channel, integer, valid)
