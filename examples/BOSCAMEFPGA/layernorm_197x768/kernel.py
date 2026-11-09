"""
Standalone Triton kernel for FPGA case 'kernel.py' (model 'layernorm_197x768').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/layernorm_197x768/kernel.py
before linking launch.c.
"""

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
    """LayerNorm over the final (hidden) dimension, with affine parameters.

    ``X``/``Out`` are row-major ``[ROWS, WIDTH]`` and ``Weight``/``Bias`` are
    ``[WIDTH]``.  ``BLOCK`` is the next power of two >= ``WIDTH``; the lanes in
    ``WIDTH..BLOCK`` are masked out of every access, so they load ``0`` (X),
    ``1`` (Weight) and ``0`` (Bias), and the reductions divide by ``WIDTH``
    rather than by ``BLOCK``.
    """
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    valid = channel < WIDTH
    x = tl.load(X + row * WIDTH + channel, valid, other=0.0).to(tl.float32)
    weight = tl.load(Weight + channel, valid, other=1.0).to(tl.float32)
    bias = tl.load(Bias + channel, valid, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / WIDTH
    centered = tl.where(valid, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / WIDTH
    out = centered * tl.rsqrt(variance + EPS) * weight + bias
    tl.store(Out + row * WIDTH + channel, out, valid)
