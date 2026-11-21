import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _pad_circular_1d_kernel(
    x_ptr,
    out_ptr,
    W,
    Wo,
    plw,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    wblk = tl.program_id(1)
    w = wblk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = w < Wo
    # circular wrap of the last dim: each coordinate wraps at most once because
    # |pad| <= W, so a single compare-select correction replaces integer modulo.
    sw = w - plw
    sw = tl.where(sw < 0, sw + W, sw)
    sw = tl.where(sw >= W, sw - W, sw)
    src = b * W + sw
    out_off = b * Wo + w
    vals = tl.load(x_ptr + src, mask=mask)
    tl.store(out_ptr + out_off, vals, mask=mask)


@triton.jit
def _pad_circular_2d_kernel(
    x_ptr,
    out_ptr,
    R,
    H,
    W,
    Ho,
    Wo,
    plh,
    plw,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    rblk = tl.program_id(0)
    wblk = tl.program_id(1)
    rows = rblk * BLOCK_H + tl.arange(0, BLOCK_H)  # over R = B * Ho
    w = wblk * BLOCK_W + tl.arange(0, BLOCK_W)

    b = rows // Ho
    h = rows % Ho
    sh = h - plh
    sh = tl.where(sh < 0, sh + H, sh)
    sh = tl.where(sh >= H, sh - H, sh)
    src_base = (b * H + sh) * W  # [BLOCK_H]

    sw = w - plw
    sw = tl.where(sw < 0, sw + W, sw)
    sw = tl.where(sw >= W, sw - W, sw)  # [BLOCK_W]
    src = src_base[:, None] + sw[None, :]
    out_off = rows[:, None] * Wo + w[None, :]
    mask = (rows[:, None] < R) & (w[None, :] < Wo)
    vals = tl.load(x_ptr + src, mask=mask)
    tl.store(out_ptr + out_off, vals, mask=mask)


@triton.jit
def _pad_circular_3d_kernel(
    x_ptr,
    out_ptr,
    R,
    D,
    H,
    W,
    Do,
    Ho,
    Wo,
    pld,
    plh,
    plw,
    BLOCK_W: tl.constexpr,
):
    # rows enumerate the collapsed (batch, Do, Ho) planes; each program handles
    # one output row of width Wo. Depth/height wrap is scalar per program, width
    # is vectorized -- the scalar-hoist pattern that won for the 2D case.
    row = tl.program_id(0)
    wblk = tl.program_id(1)

    b = row // (Do * Ho)
    rem = row % (Do * Ho)
    d = rem // Ho
    h = rem % Ho

    sd = d - pld
    sd = tl.where(sd < 0, sd + D, sd)
    sd = tl.where(sd >= D, sd - D, sd)
    sh = h - plh
    sh = tl.where(sh < 0, sh + H, sh)
    sh = tl.where(sh >= H, sh - H, sh)
    src_base = ((b * D + sd) * H + sh) * W

    w = wblk * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (row < R) & (w < Wo)
    sw = w - plw
    sw = tl.where(sw < 0, sw + W, sw)
    sw = tl.where(sw >= W, sw - W, sw)
    src = src_base + sw
    out_off = row * Wo + w
    vals = tl.load(x_ptr + src, mask=mask)
    tl.store(out_ptr + out_off, vals, mask=mask)


def _pad_circular(x: torch.Tensor, pad):
    """aten::_pad_circular: circular padding of the last 1-3 dims (batch dim kept).

    Padding semantics follow aten (PadNd.cpp): non-positive padding first crops
    the input (``narrow``), then the remaining non-negative pads wrap within the
    *cropped* tensor; the wrap guard (each pad <= the cropped dim size, i.e. at
    most one wrap) reproduces aten's "Padding value causes wrapping around more
    than once." error. Upstream FlagGems wrapped negative pads within the
    original tensor instead; this port matches aten because that is the value
    oracle. Degenerate mixes where aten reads uninitialized memory (crop empties
    the dim but the output dim stays positive) keep upstream's
    "negative padding removed the whole dimension" error instead of UB.
    """
    logger.debug("GEMS _PAD_CIRCULAR")

    ndim = x.dim()
    if len(pad) % 2 != 0:
        raise ValueError("padding length must be even")
    npad = len(pad) // 2
    if npad < 1 or npad > 3:
        raise ValueError(
            "flag_gems _pad_circular supports padding the last 1, 2 or 3 dimensions"
        )
    if ndim <= npad:
        raise ValueError("input must have at least one non-padded (batch) dimension")

    x = x.contiguous()
    in_shape = list(x.shape)
    out_shape = list(in_shape)
    pad_left = [0] * ndim
    for i in range(npad):
        d = ndim - 1 - i
        pl = int(pad[2 * i])
        pr = int(pad[2 * i + 1])
        new_size = in_shape[d] + pl + pr
        if new_size < 0:
            raise ValueError(
                "negative padding removed the whole dimension "
                f"{d}: input {in_shape[d]} pad ({pl}, {pr})"
            )
        # aten crops by the negative amounts first, then circular-pads the
        # cropped dim with the remaining non-negative amounts.
        crop_l = max(-pl, 0)
        crop_r = max(-pr, 0)
        cropped = in_shape[d] - crop_l - crop_r
        if cropped == 0 and new_size > 0:
            # aten produces uninitialized memory here; keep upstream's error.
            raise ValueError(
                "negative padding removed the whole dimension "
                f"{d}: input {in_shape[d]} pad ({pl}, {pr})"
            )
        if cropped > 0:
            if crop_l or crop_r:
                x = x.narrow(d, crop_l, cropped)
            mpl = max(pl, 0)
            mpr = max(pr, 0)
            if mpl > cropped or mpr > cropped:
                raise ValueError("Padding value causes wrapping around more than once.")
        out_shape[d] = new_size
        pad_left[d] = max(pl, 0)

    # Cropping (narrow) leaves a non-contiguous view; the kernels index a
    # flat contiguous buffer, so materialize one.
    x = x.contiguous()
    in_shape = list(x.shape)

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    total = 1
    for s in out_shape:
        total *= s
    if total == 0:
        return out

    # BLOCK_SIZE=1024 for the 1D-pad path: on this bandwidth-bound copy, many small
    # resident programs beat fewer large tiles; 2048 regressed ~5% (lower occupancy).
    BLOCK_SIZE = 1024

    if npad == 1:
        W = in_shape[-1]
        Wo = out_shape[-1]
        plw = pad_left[-1]
        B = 1
        for s in in_shape[:-1]:
            B *= s
        grid = (B, triton.cdiv(Wo, BLOCK_SIZE))
        _pad_circular_1d_kernel[grid](x, out, W, Wo, plw, BLOCK_SIZE=BLOCK_SIZE)
    elif npad == 2:
        H = in_shape[-2]
        W = in_shape[-1]
        Ho = out_shape[-2]
        Wo = out_shape[-1]
        plh = pad_left[-2]
        plw = pad_left[-1]
        B = 1
        for s in in_shape[:-2]:
            B *= s
        R = B * Ho
        # narrow rows waste lanes in a 1024-wide 1D block; a 2D tile fills warps.
        if Wo >= 2048:
            BLOCK_H, BLOCK_W = 1, 1024
        elif Wo >= 256:
            BLOCK_H, BLOCK_W = 4, 256
        elif Wo >= 64:
            BLOCK_H, BLOCK_W = 8, 128
        else:
            BLOCK_H, BLOCK_W = 32, 32
        grid = (triton.cdiv(R, BLOCK_H), triton.cdiv(Wo, BLOCK_W))
        _pad_circular_2d_kernel[grid](
            x,
            out,
            R,
            H,
            W,
            Ho,
            Wo,
            plh,
            plw,
            BLOCK_H=BLOCK_H,
            BLOCK_W=BLOCK_W,
        )
    else:  # npad == 3
        D = in_shape[-3]
        H = in_shape[-2]
        W = in_shape[-1]
        Do = out_shape[-3]
        Ho = out_shape[-2]
        Wo = out_shape[-1]
        pld = pad_left[-3]
        plh = pad_left[-2]
        plw = pad_left[-1]
        B = 1
        for s in in_shape[:-3]:
            B *= s
        R = B * Do * Ho
        BLOCK_W = 1024
        grid = (R, triton.cdiv(Wo, BLOCK_W))
        _pad_circular_3d_kernel[grid](
            x,
            out,
            R,
            D,
            H,
            W,
            Do,
            Ho,
            Wo,
            pld,
            plh,
            plw,
            BLOCK_W=BLOCK_W,
        )
    return out
