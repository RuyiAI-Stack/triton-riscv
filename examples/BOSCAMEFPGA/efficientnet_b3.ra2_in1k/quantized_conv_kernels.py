"""Triton stages for convolution -> im2col -> W8A8 AME GEMM pipelines."""

import triton
import triton.language as tl


@triton.jit
def quantize_weight(
    Weight,
    QWeight,
    WeightScale,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    KPAD: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, KPAD)
    value = tl.load(Weight + row * WIDTH + k, k < WIDTH, other=0.0).to(tl.float32)
    maximum = tl.max(tl.abs(value), axis=0)
    scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
    scaled = tl.minimum(tl.maximum(tl.div_rn(value, scale), -127.0), 127.0)
    base = scaled.to(tl.int32)
    fraction = scaled - base.to(tl.float32)
    adjustment = tl.where(
        tl.abs(fraction) >= 0.5,
        tl.where(scaled >= 0, 1, -1),
        0,
    )
    quantized = tl.where(k < WIDTH, base + adjustment, 0).to(tl.int8)
    tl.store(QWeight + row * KPAD + k, quantized)
    tl.store(WeightScale + row, scale)


@triton.jit
def quantize_im2col(
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
    """Prepare one complete patch tile with one load per logical input tap.

    The bounded K tuple preserves each FP32 value until its per-patch scale is
    known. Quantization keeps the original max order and ties-away rounding;
    the adapter initializes QPatch padding once per public call. Logical boundary
    taps and output-tail rows are written as zero, with zero-row scale one.
    """
    group = tl.program_id(2)
    if DEPTHWISE:
        K: tl.constexpr = KH * KW
    else:
        K: tl.constexpr = C * KH * KW
    tl.static_assert(K <= KPAD)
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
            value = values[r]
            scaled = tl.minimum(tl.maximum(tl.div_rn(value, scale), -127.0), 127.0)
            base = scaled.to(tl.int32)
            fraction = scaled - base.to(tl.float32)
            adjustment = tl.where(
                tl.abs(fraction) >= 0.5,
                tl.where(scaled >= 0, 1, -1),
                0,
            )
            tl.store(QPatch + lane * KPAD + r, (base + adjustment).to(tl.int8))
        tl.store(PatchScale + lane, scale)


@triton.jit
def conv_gemm_i8(
    QWeight,
    QPatch,
    Acc,
    M: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DEPTHWISE: tl.constexpr,
):
    group = tl.program_id(2)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    k = tl.arange(0, BK)
    if DEPTHWISE:
        weight_rows = tl.full((BM,), group, tl.int32)
    else:
        weight_rows = rows
    a = tl.load(
        QWeight + weight_rows[:, None] * K + k[None, :],
        rows[:, None] < M,
        other=0,
    )
    b = tl.load(QPatch + cols[None, :] * K + k[:, None])
    accumulator = tl.zeros((BM, BN), dtype=tl.int32)
    accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
    local_rows = rows - tl.program_id(0) * BM
    tl.store(
        Acc + local_rows[:, None] * BN + cols[None, :], accumulator, rows[:, None] < M
    )


@triton.jit
def conv_dequantize_bias(
    Acc,
    PatchScale,
    WeightScale,
    Bias,
    Out,
    OC: tl.constexpr,
    P: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    TILE: tl.constexpr,
    DEPTHWISE: tl.constexpr,
):
    oc = tl.program_id(0)
    patch_block = tl.program_id(1)
    _m_block = tl.program_id(2)
    position = patch_block * TILE + tl.arange(0, TILE)
    if not DEPTHWISE:
        acc_row = oc % BM
    else:
        acc_row = 0
    accumulator = tl.load(Acc + acc_row * BN + tl.arange(0, TILE))
    patch_scale = tl.load(PatchScale + tl.arange(0, TILE))
    weight_scale = tl.load(WeightScale + oc)
    bias = tl.load(Bias + oc)
    value = accumulator.to(tl.float32) * weight_scale
    value = value * patch_scale + bias
    tl.store(Out + oc * P + position, value, (oc < OC) & (position < P))


def conv2d_im2col_w8a8(case):
    """Bind an ordinary convolution to its four W8A8 MMA stages."""
    if case["family"] != "conv2d_f32":
        raise ValueError("conv2d_im2col_w8a8 requires ordinary convolution")
    return _stage_configs(case, depthwise=False)


def dwconv2d_im2col_w8a8(case):
    """Bind a depthwise convolution to its four W8A8 MMA stages."""
    if case["family"] != "dwconv2d_f32":
        raise ValueError("dwconv2d_im2col_w8a8 requires depthwise convolution")
    return _stage_configs(case, depthwise=True)


def _stage_configs(case, depthwise):
    """Return the independently exported raw stages for one public conv."""
    spec = case["constexprs"]
    rows = spec["C"] if depthwise else spec["OC"]
    width = spec["KH"] * spec["KW"] * (1 if depthwise else spec["C"])
    positions = spec["OH"] * spec["OW"]
    tile = 64
    kpad = 64
    bm = 1 if depthwise else 16
    spatial_tiles = triton.cdiv(positions, tile)
    m_tiles = 1 if depthwise else triton.cdiv(rows, bm)
    groups = spec["C"] if depthwise else 1
    x_shape = [1, spec["C"], spec["H"], spec["W"]]
    return [
        {
            "name": "weight_quantize",
            "kernel": quantize_weight,
            "arguments": [
                {"name": "Weight", "rank": 2, "dtype": "f32"},
                {"name": "QWeight", "rank": 2, "dtype": "i8"},
                {"name": "WeightScale", "rank": 1, "dtype": "f32"},
            ],
            "shapes": [[rows, width], [rows, kpad], [rows]],
            "constexprs": {"ROWS": rows, "WIDTH": width, "KPAD": kpad},
            "grid": [rows, 1, 1],
        },
        {
            "name": "im2col_quantize",
            "kernel": quantize_im2col,
            "arguments": [
                {"name": "X", "rank": 4, "dtype": "f32"},
                {"name": "QPatch", "rank": 2, "dtype": "i8"},
                {"name": "PatchScale", "rank": 1, "dtype": "f32"},
            ],
            "shapes": [x_shape, [tile, kpad], [tile]],
            "constexprs": {
                **{
                    key: spec[key]
                    for key in (
                        "H",
                        "W",
                        "C",
                        "KH",
                        "KW",
                        "SH",
                        "SW",
                        "PH",
                        "PW",
                        "OH",
                        "OW",
                    )
                },
                "DEPTHWISE": depthwise,
                "KPAD": kpad,
                "TILE": tile,
            },
            "grid": [spatial_tiles, 1, groups],
        },
        {
            "name": "gemm_i8",
            "kernel": conv_gemm_i8,
            "arguments": [
                {"name": "QWeight", "rank": 2, "dtype": "i8"},
                {"name": "QPatch", "rank": 2, "dtype": "i8"},
                {"name": "Acc", "rank": 2, "dtype": "i32"},
            ],
            "shapes": [[rows, kpad], [tile, kpad], [bm, tile]],
            "constexprs": {
                "M": rows if not depthwise else 1,
                "K": kpad,
                "BM": bm,
                "BN": tile,
                "BK": kpad,
                "DEPTHWISE": depthwise,
            },
            "grid": [m_tiles, 1, groups],
        },
        {
            "name": "dequantize_bias",
            "kernel": conv_dequantize_bias,
            "arguments": [
                {"name": "Acc", "rank": 2, "dtype": "i32"},
                {"name": "PatchScale", "rank": 1, "dtype": "f32"},
                {"name": "WeightScale", "rank": 1, "dtype": "f32"},
                {"name": "Bias", "rank": 1, "dtype": "f32"},
                {"name": "Out", "rank": 2, "dtype": "f32"},
            ],
            "shapes": [[bm, tile], [tile], [rows], [rows], [rows, positions]],
            "constexprs": {
                "OC": rows,
                "P": positions,
                "BM": bm,
                "BN": tile,
                "TILE": tile,
                "DEPTHWISE": depthwise,
            },
            "grid": [rows, spatial_tiles, m_tiles],
        },
    ]
