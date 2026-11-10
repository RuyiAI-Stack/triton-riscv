"""Quantize/dequantize stages for the public W8A8 classifier."""

import triton
import triton.language as tl


@triton.jit
def quantize_rows(X, Q, Scale, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + k, k < WIDTH, other=0.0)
    maximum = tl.max(tl.abs(x), axis=0)
    scale = tl.where(maximum == 0.0, 1.0, tl.div_rn(maximum, 127.0))
    value = tl.minimum(tl.maximum(tl.div_rn(x, scale), -127.0), 127.0)
    base = value.to(tl.int32)
    fraction = value - base.to(tl.float32)
    adjustment = tl.where(tl.abs(fraction) >= 0.5, tl.where(value >= 0.0, 1, -1), 0)
    tl.store(Q + row * WIDTH + k, (base + adjustment).to(tl.int8), k < WIDTH)
    tl.store(Scale + row, scale)


@triton.jit
def zero_acc(Acc, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Acc + i, tl.full((BLOCK,), 0, tl.int32), i < COUNT)


@triton.jit
def dequantize_bias(
    Acc,
    RowScale,
    WeightScale,
    Bias,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(Acc + row * N + col, col < N, other=0).to(tl.float32)
    rs = tl.load(RowScale + row)
    ws = tl.load(WeightScale + col, col < N, other=0.0)
    bias = tl.load(Bias + col, col < N, other=0.0)
    tl.store(Out + row * N + col, (acc * ws) * rs + bias, col < N)
