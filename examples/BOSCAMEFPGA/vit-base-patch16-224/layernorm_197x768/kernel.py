"""Standalone Triton source for layernorm_197x768."""

import triton
import triton.language as tl


@triton.jit
def layernorm(
    X,
    Weight,
    Bias,
    Out,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    """LayerNorm over the final (hidden) dimension, with affine parameters."""
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    valid = channel < WIDTH
    x = tl.load(X + row * WIDTH + channel, valid, other=0.0).to(tl.float32)
    weight = tl.load(Weight + channel, valid, other=1.0).to(tl.float32)
    bias = tl.load(Bias + channel, valid, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / WIDTH
    # Padding must stay zero after subtraction, or it changes the variance.
    centered = tl.where(valid, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / WIDTH
    out = centered * tl.rsqrt(variance + EPS) * weight + bias
    tl.store(Out + row * WIDTH + channel, out, valid)
