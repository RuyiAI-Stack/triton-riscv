"""Independent integration prototypes; not bound into the production graph.

These connectors preserve the existing 37 packed INT8 pointwise kernels.
They require export, actual-IR checks and later board validation before use.
"""

import triton
import triton.language as tl


@triton.jit
def quantize_pack_nchw(
    X, QPacked, ActivationScale, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr
):
    """Conservative scalar baseline: Q[HW,C], Scale[HW], one pixel/program.

    The current contract exporter rejects the strided vector gather and the
    rectangular reduction/broadcast variant. This is correctness scaffolding,
    not a proposed performance endpoint.
    """
    position = tl.program_id(0)
    maximum = 0.0
    for channel in range(C):
        value = tl.load(X + channel * H * W + position)
        maximum = tl.maximum(maximum, tl.abs(value))
    scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
    for channel in range(C):
        value = tl.load(X + channel * H * W + position)
        ratio = tl.minimum(tl.maximum(tl.div_rn(value, scale), -127.0), 127.0)
        base = ratio.to(tl.int32)
        fraction = ratio - base.to(tl.float32)
        adjustment = tl.where(tl.abs(fraction) >= 0.5, tl.where(ratio >= 0, 1, -1), 0)
        tl.store(QPacked + position * C + channel, (base + adjustment).to(tl.int8))
    tl.store(ActivationScale + position, scale)


@triton.jit
def dequantize_unpack_bias(
    Acc,
    ActivationScale,
    WeightScale,
    Bias,
    Out,
    OC: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    """Scalar mapping baseline, NCHW = Acc[HW,OC]*WS[OC]*AS[HW]+Bias[OC]."""
    position = tl.program_id(0)
    channel = tl.program_id(1)
    acc = tl.load(Acc + position * OC + channel).to(tl.float32)
    ws = tl.load(WeightScale + channel)
    act = tl.load(ActivationScale + position)
    bias = tl.load(Bias + channel)
    value = acc * ws
    value = value * act + bias
    tl.store(Out + channel * H * W + position, value)


@triton.jit
def sigmoid(X, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    value = tl.load(X + index, valid, other=0.0)
    result = 1.0 / (1.0 + tl.exp(-value))
    tl.store(Out + index, result, valid)


@triton.jit
def se_broadcast_multiply(
    X, Gate, Out, C: tl.constexpr, SPATIAL: tl.constexpr, BLOCK: tl.constexpr
):
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = tl.program_id(1)
    valid = position < SPATIAL
    index = channel * SPATIAL + position
    value = tl.load(X + index, valid, other=0.0)
    gate = tl.load(Gate + channel)
    tl.store(Out + index, value * gate, valid)


@triton.jit
def silu(X, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    value = tl.load(X + index, valid, other=0.0)
    tl.store(Out + index, value / (1.0 + tl.exp(-value)), valid)


@triton.jit
def zero_accumulator(Acc, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Acc + index, 0, index < COUNT)


@triton.jit
def classifier_bias_add(X, Bias, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    value = tl.load(X + index, valid, other=0.0)
    bias = tl.load(Bias + index, valid, other=0.0)
    tl.store(Out + index, value + bias, valid)
