"""Independent tile-level convolution preparation prototypes; production untouched."""

import triton
import triton.language as tl


@triton.jit
def tile_im2col_quantize(
    X,
    QPatch,
    PatchScale,
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
    positions = tl.program_id(0) * TILE + tl.arange(0, TILE)
    group = tl.program_id(2)
    if DEPTHWISE:
        K: tl.constexpr = KH * KW
    else:
        K: tl.constexpr = C * KH * KW
    maximum = tl.full((TILE,), 0.0, tl.float32)
    for r in range(K):
        if DEPTHWISE:
            ic = group
        else:
            ic = r // (KH * KW)
        kh = r // KW % KH
        kw = r % KW
        ih = positions // OW * SH - PH + kh
        iw = positions % OW * SW - PW + kw
        valid = (positions < OH * OW) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        value = tl.load(X + ic * H * W + ih * W + iw, valid, 0.0)
        maximum = tl.maximum(maximum, tl.abs(value))
    scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
    for r in range(KPAD):
        integer = tl.full((TILE,), 0, tl.int32)
        if r < K:
            if DEPTHWISE:
                ic = group
            else:
                ic = r // (KH * KW)
            kh = r // KW % KH
            kw = r % KW
            ih = positions // OW * SH - PH + kh
            iw = positions % OW * SW - PW + kw
            valid = (positions < OH * OW) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            value = tl.load(X + ic * H * W + ih * W + iw, valid, 0.0)
            scaled = tl.minimum(tl.maximum(tl.div_rn(value, scale), -127.0), 127.0)
            base = scaled.to(tl.int32)
            fraction = scaled - base.to(tl.float32)
            adjustment = tl.where(
                tl.abs(fraction) >= 0.5, tl.where(scaled >= 0, 1, -1), 0
            )
            integer = base + adjustment
        tl.store(QPatch + tl.arange(0, TILE) * KPAD + r, integer.to(tl.int8))
    tl.store(PatchScale + tl.arange(0, TILE), scale)


@triton.jit
def cached_tile_im2col_quantize(
    X,
    QPatch,
    PatchScale,
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
            tl.store(FPatch + r, value)
        k = tl.arange(0, KPAD)
        value = tl.load(FPatch + k)
        maximum = tl.max(tl.abs(value), axis=0)
        scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
        scaled = tl.minimum(tl.maximum(tl.div_rn(value, scale), -127.0), 127.0)
        base = scaled.to(tl.int32)
        fraction = scaled - base.to(tl.float32)
        adjustment = tl.where(tl.abs(fraction) >= 0.5, tl.where(scaled >= 0, 1, -1), 0)
        tl.store(QPatch + lane * KPAD + k, (base + adjustment).to(tl.int8))
        tl.store(PatchScale + lane, scale)


@triton.jit
def cached_scalar_tile_im2col_quantize(
    X,
    QPatch,
    PatchScale,
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
        maximum = 0.0
        values = ()
        for r in tl.static_range(K):
            value = 0.0
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
            maximum = tl.maximum(maximum, tl.abs(value))
            values += (value,)
        scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
        for r in tl.static_range(KPAD):
            integer = 0
            if r < K:
                value = values[r]
                scaled = tl.minimum(tl.maximum(tl.div_rn(value, scale), -127.0), 127.0)
                base = scaled.to(tl.int32)
                fraction = scaled - base.to(tl.float32)
                adjustment = tl.where(
                    tl.abs(fraction) >= 0.5, tl.where(scaled >= 0, 1, -1), 0
                )
                integer = base + adjustment
            tl.store(QPatch + lane * KPAD + r, integer.to(tl.int8))
        tl.store(PatchScale + lane, scale)


@triton.jit
def cached_precleared_tile_im2col_quantize(
    X,
    QPatch,
    PatchScale,
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
    # Internal adapter guarantees QPatch padded columns are zero before any tile.
    group = tl.program_id(2)
    if DEPTHWISE:
        K: tl.constexpr = KH * KW
    else:
        K: tl.constexpr = C * KH * KW
    for lane in range(TILE):
        position = tl.program_id(0) * TILE + lane
        maximum = 0.0
        values = ()
        for r in tl.static_range(K):
            value = 0.0
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
            maximum = tl.maximum(maximum, tl.abs(value))
            values += (value,)
        scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
        for r in tl.static_range(K):
            integer = 0
            if r < K:
                value = values[r]
                scaled = tl.minimum(tl.maximum(tl.div_rn(value, scale), -127.0), 127.0)
                base = scaled.to(tl.int32)
                fraction = scaled - base.to(tl.float32)
                adjustment = tl.where(
                    tl.abs(fraction) >= 0.5, tl.where(scaled >= 0, 1, -1), 0
                )
                integer = base + adjustment
            tl.store(QPatch + lane * KPAD + r, integer.to(tl.int8))
        tl.store(PatchScale + lane, scale)


@triton.jit
def tile_vector_quantize_narrow(
    FPatch,
    QPatch,
    PatchScale,
    WIDTH: tl.constexpr,
    ACTIVE: tl.constexpr,
    KPAD: tl.constexpr,
    TILE: tl.constexpr,
):
    for lane in range(TILE):
        k = tl.arange(0, ACTIVE)
        value = tl.load(FPatch + lane * KPAD + k)
        maximum = tl.max(tl.abs(value), axis=0)
        scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
        scaled = tl.minimum(tl.maximum(tl.div_rn(value, scale), -127.0), 127.0)
        base = scaled.to(tl.int32)
        fraction = scaled - base.to(tl.float32)
        adjustment = tl.where(tl.abs(fraction) >= 0.5, tl.where(scaled >= 0, 1, -1), 0)
        tl.store(QPatch + lane * KPAD + k, (base + adjustment).to(tl.int8))
        for padding in range(ACTIVE, KPAD, ACTIVE):
            tl.store(QPatch + lane * KPAD + padding + k, tl.full((ACTIVE,), 0, tl.int8))
        tl.store(PatchScale + lane, scale)


@triton.jit
def tile_gather_im2col(
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
def tile_vector_quantize(
    FPatch, QPatch, PatchScale, KPAD: tl.constexpr, TILE: tl.constexpr
):
    for lane in range(TILE):
        k = tl.arange(0, KPAD)
        value = tl.load(FPatch + lane * KPAD + k)
        maximum = tl.max(tl.abs(value), axis=0)
        scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
        scaled = tl.minimum(tl.maximum(tl.div_rn(value, scale), -127.0), 127.0)
        base = scaled.to(tl.int32)
        fraction = scaled - base.to(tl.float32)
        adjustment = tl.where(tl.abs(fraction) >= 0.5, tl.where(scaled >= 0, 1, -1), 0)
        tl.store(QPatch + lane * KPAD + k, (base + adjustment).to(tl.int8))
        tl.store(PatchScale + lane, scale)


@triton.jit
def batched_scalar_im2col_quantize(
    X,
    QPatch,
    PatchScale,
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
        maximum = 0.0
        for r in range(K):
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
                maximum = tl.maximum(maximum, tl.abs(value))
        scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
        for r in range(KPAD):
            integer = 0
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
                    scaled = tl.minimum(
                        tl.maximum(tl.div_rn(value, scale), -127.0), 127.0
                    )
                    base = scaled.to(tl.int32)
                    fraction = scaled - base.to(tl.float32)
                    adjustment = tl.where(
                        tl.abs(fraction) >= 0.5, tl.where(scaled >= 0, 1, -1), 0
                    )
                    integer = base + adjustment
            tl.store(QPatch + lane * KPAD + r, integer.to(tl.int8))
        tl.store(PatchScale + lane, scale)


@triton.jit
def register_tile_im2col_quantize(
    X,
    QPatch,
    PatchScale,
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
        k = tl.arange(0, KPAD)
        position = tl.program_id(0) * TILE + lane
        patch = tl.full((KPAD,), 0.0, tl.float32)
        for r in range(K):
            value = 0.0
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
            patch = tl.where(k == r, value, patch)
        maximum = tl.max(tl.abs(patch), axis=0)
        scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
        scaled = tl.minimum(tl.maximum(tl.div_rn(patch, scale), -127.0), 127.0)
        base = scaled.to(tl.int32)
        fraction = scaled - base.to(tl.float32)
        adjustment = tl.where(tl.abs(fraction) >= 0.5, tl.where(scaled >= 0, 1, -1), 0)
        tl.store(QPatch + lane * KPAD + k, (base + adjustment).to(tl.int8))
        tl.store(PatchScale + lane, scale)


@triton.jit
def batched_vector_padding_im2col_quantize(
    X,
    QPatch,
    PatchScale,
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
        maximum = 0.0
        for r in range(K):
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
                maximum = tl.maximum(maximum, tl.abs(value))
        scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
        tl.store(
            QPatch + lane * KPAD + tl.arange(0, KPAD), tl.full((KPAD,), 0, tl.int8)
        )
        for r in range(K):
            integer = 0
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
                    scaled = tl.minimum(
                        tl.maximum(tl.div_rn(value, scale), -127.0), 127.0
                    )
                    base = scaled.to(tl.int32)
                    fraction = scaled - base.to(tl.float32)
                    adjustment = tl.where(
                        tl.abs(fraction) >= 0.5, tl.where(scaled >= 0, 1, -1), 0
                    )
                    integer = base + adjustment
            tl.store(QPatch + lane * KPAD + r, integer.to(tl.int8))
        tl.store(PatchScale + lane, scale)
