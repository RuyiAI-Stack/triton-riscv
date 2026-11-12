"""Diagnostic-only split of the production fused im2col/activation quantization."""

import triton
import triton.language as tl


@triton.jit
def pure_im2col(
    X,
    FPatch,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    DEPTHWISE: tl.constexpr,
    KPAD: tl.constexpr,
    TILE: tl.constexpr,
):
    group = tl.program_id(2)
    if DEPTHWISE:
        K: tl.constexpr = KH * KW
    else:
        K: tl.constexpr = C * KH * KW
    for lane in range(TILE):
        position = tl.program_id(0) * TILE + lane
        for r in range(KPAD):
            value = 0.0
            if r < K:
                if DEPTHWISE:
                    ic = group
                else:
                    ic = r // (KH * KW)
                kh = r // KW % KH
                kw = r % KW
                ih = position // OW * SH - PH + kh
                iw = position % OW * SW - PW + kw
                if (position < OH * OW) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W):
                    value = tl.load(X + ic * H * W + ih * W + iw)
            tl.store(FPatch + lane * KPAD + r, value)


@triton.jit
def activation_quantize(
    FPatch,
    QPatch,
    PatchScale,
    WIDTH: tl.constexpr,
    KPAD: tl.constexpr,
    TILE: tl.constexpr,
):
    for lane in range(TILE):
        maximum = 0.0
        for r in range(WIDTH):
            value = tl.load(FPatch + lane * KPAD + r).to(tl.float32)
            maximum = tl.maximum(maximum, tl.abs(value))
        scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
        for r in range(KPAD):
            integer = 0
            if r < WIDTH:
                value = tl.load(FPatch + lane * KPAD + r).to(tl.float32)
                scaled = tl.minimum(tl.maximum(tl.div_rn(value, scale), -127.0), 127.0)
                base = scaled.to(tl.int32)
                fraction = scaled - base.to(tl.float32)
                adjustment = tl.where(
                    tl.abs(fraction) >= 0.5,
                    tl.where(scaled >= 0, 1, -1),
                    0,
                )
                integer = base + adjustment
            tl.store(QPatch + lane * KPAD + r, integer.to(tl.int8))
        tl.store(PatchScale + lane, scale)
