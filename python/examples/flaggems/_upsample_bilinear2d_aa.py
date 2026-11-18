import math

import torch
import triton
import triton.language as tl


@triton.jit
def upsample_bilinear2d_aa_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    ow = (pid_x * BLOCK_X + tl.arange(0, BLOCK_X)) % OW
    oh = (pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)) % OH

    # Bilinear interpolation uses 2x2 support window
    support_w = 1.0
    support_h = 1.0

    # _compute_weights_span
    center_w = (ow + 0.5) * reciprocal_scale_w
    center_h = (oh + 0.5) * reciprocal_scale_h
    span_start_w = tl.maximum(center_w - support_w + 0.5, 0).to(tl.int32)
    span_start_h = tl.maximum(center_h - support_h + 0.5, 0).to(tl.int32)
    span_size_w = (tl.minimum(center_w + support_w + 0.5, IW) - span_start_w).to(
        tl.int32
    )
    span_size_h = (tl.minimum(center_h + support_h + 0.5, IH) - span_start_h).to(
        tl.int32
    )
    start_minus_center_w = span_start_w - center_w
    start_minus_center_h = span_start_h - center_h

    # Bilinear weights: linear interpolation in each dimension
    # For y direction: y0 = (ih - center_h), y1 = (ih + 1 - center_h)
    wy0 = 1.0 - tl.abs(start_minus_center_h + 0.5)
    wy1 = 1.0 - tl.abs(start_minus_center_h + 1.5)
    weight_y0 = tl.where(
        0 < span_size_h,
        tl.maximum(wy0, 0),
        0,
    )
    weight_y1 = tl.where(
        1 < span_size_h,
        tl.maximum(wy1, 0),
        0,
    )
    # Normalize y weights
    weight_y_total = weight_y0 + weight_y1
    weight_y_total = tl.where(weight_y_total != 0, weight_y_total, 1)
    weight_y0 = weight_y0 / weight_y_total
    weight_y1 = weight_y1 / weight_y_total

    # For x direction: x0 = (iw - center_w), x1 = (iw + 1 - center_w)
    wx0 = 1.0 - tl.abs(start_minus_center_w + 0.5)
    wx1 = 1.0 - tl.abs(start_minus_center_w + 1.5)
    weight_x0 = tl.where(
        0 < span_size_w,
        tl.maximum(wx0, 0),
        0,
    )
    weight_x1 = tl.where(
        1 < span_size_w,
        tl.maximum(wx1, 0),
        0,
    )
    # Normalize x weights
    weight_x_total = weight_x0 + weight_x1
    weight_x_total = tl.where(weight_x_total != 0, weight_x_total, 1)
    weight_x0 = weight_x0 / weight_x_total
    weight_x1 = weight_x1 / weight_x_total

    mask_y0 = span_start_h[:, None] + 0 < IH
    mask_y1 = span_start_h[:, None] + 1 < IH
    mask_x0 = span_start_w[None, :] + 0 < IW
    mask_x1 = span_start_w[None, :] + 1 < IW

    for n in range(0, N, 1):
        for c in range(0, C, 1):
            offset_base = (
                (n * C + c) * IH + span_start_h[:, None]
            ) * IW + span_start_w[None, :]

            data00 = tl.load(
                ptr_i + (offset_base + 0 * IW + 0),
                mask=(mask_y0 & mask_x0),
                other=0,
            )
            data01 = tl.load(
                ptr_i + (offset_base + 0 * IW + 1),
                mask=(mask_y0 & mask_x1),
                other=0,
            )
            data10 = tl.load(
                ptr_i + (offset_base + 1 * IW + 0),
                mask=(mask_y1 & mask_x0),
                other=0,
            )
            data11 = tl.load(
                ptr_i + (offset_base + 1 * IW + 1),
                mask=(mask_y1 & mask_x1),
                other=0,
            )

            # Bilinear interpolation: first in x direction, then in y direction
            row0 = data00 * weight_x0[None, :] + data01 * weight_x1[None, :]
            row1 = data10 * weight_x0[None, :] + data11 * weight_x1[None, :]
            result = row0 * weight_y0[:, None] + row1 * weight_y1[:, None]

            offset_o = ((n * C + c) * OH + oh[:, None]) * OW + ow[None, :]
            tl.store(ptr_o + offset_o, result)


@triton.jit
def general_interpolate_bilinear2d_aa_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    MAX_INTERPOLATE_H: tl.constexpr,
    MAX_INTERPOLATE_W: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    ow = (pid_x * BLOCK_X + tl.arange(0, BLOCK_X)) % OW
    oh = (pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)) % OH

    support_w = reciprocal_scale_w if reciprocal_scale_w >= 1.0 else 1.0
    support_h = reciprocal_scale_h if reciprocal_scale_h >= 1.0 else 1.0

    center_w = (ow + 0.5) * reciprocal_scale_w
    center_h = (oh + 0.5) * reciprocal_scale_h
    span_start_w = tl.maximum(center_w - support_w + 0.5, 0).to(tl.int32)
    span_start_h = tl.maximum(center_h - support_h + 0.5, 0).to(tl.int32)
    span_size_w = (tl.minimum(center_w + support_w + 0.5, IW) - span_start_w).to(
        tl.int32
    )
    span_size_h = (tl.minimum(center_h + support_h + 0.5, IH) - span_start_h).to(
        tl.int32
    )

    invscale_w = 1.0 / reciprocal_scale_w if reciprocal_scale_w >= 1.0 else 1.0
    invscale_h = 1.0 / reciprocal_scale_h if reciprocal_scale_h >= 1.0 else 1.0
    start_minus_center_w = span_start_w - center_w
    start_minus_center_h = span_start_h - center_h

    for n in range(0, N, 1):
        for c in range(0, C, 1):
            offset_base = (
                (n * C + c) * IH + span_start_h[:, None]
            ) * IW + span_start_w[None, :]
            weight_y_total = tl.zeros((BLOCK_Y,), dtype=tl.float32)
            result = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
            for y in range(0, MAX_INTERPOLATE_H, 1):
                wy = tl.abs((y + start_minus_center_h + 0.5) * invscale_h)
                weight_y = tl.where(y < span_size_h, tl.maximum(1.0 - wy, 0.0), 0.0)
                weight_y_total += weight_y
                weight_x_total = tl.zeros((BLOCK_X,), dtype=tl.float32)
                buffer = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
                for x in range(0, MAX_INTERPOLATE_W, 1):
                    wx = tl.abs((x + start_minus_center_w + 0.5) * invscale_w)
                    weight_x = tl.where(x < span_size_w, tl.maximum(1.0 - wx, 0.0), 0.0)
                    weight_x_total += weight_x
                    data = tl.load(
                        ptr_i + (offset_base + y * IW + x),
                        mask=(span_start_h[:, None] + y < IH)
                        & (span_start_w[None, :] + x < IW),
                        other=0,
                    )
                    buffer += data.to(tl.float32) * weight_x[None, :]
                weight_x_total = tl.where(weight_x_total != 0, weight_x_total, 1.0)
                result += buffer / weight_x_total[None, :] * weight_y[:, None]
            weight_y_total = tl.where(weight_y_total != 0, weight_y_total, 1.0)
            result /= weight_y_total[:, None]
            offset_o = ((n * C + c) * OH + oh[:, None]) * OW + ow[None, :]
            tl.store(ptr_o + offset_o, result.to(ptr_o.dtype.element_ty))


def bilinear_reciprocal_scale(src_size, dst_size, align_corners, scale):
    if align_corners:
        if dst_size > 1:
            return (src_size - 1) / (dst_size - 1)
        else:
            return 0
    else:
        if scale is not None and scale > 0:
            return 1.0 / scale
        else:
            return src_size / dst_size


# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
def _upsample_bilinear2d_aa(
    input: torch.Tensor,
    output_size: tuple[int],
    align_corners: bool = False,
    scales_h: float | None = None,
    scales_w: float | None = None,
):
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"

    OH, OW = output_size
    N, C, IH, IW = input.shape

    reciprocal_scale_h = bilinear_reciprocal_scale(IH, OH, align_corners, scales_h)
    reciprocal_scale_w = bilinear_reciprocal_scale(IW, OW, align_corners, scales_w)

    # allocate output
    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)

    def grid(META):
        return (
            triton.cdiv(OW, META["BLOCK_X"]),
            triton.cdiv(OH, META["BLOCK_Y"]),
        )

    BLOCK_X = 32
    BLOCK_Y = 32

    is_downsampling = reciprocal_scale_h > 1.0 or reciprocal_scale_w > 1.0
    if is_downsampling:
        max_interpolate_h = max(1, math.ceil(max(reciprocal_scale_h, 1.0)) * 2 + 1)
        max_interpolate_w = max(1, math.ceil(max(reciprocal_scale_w, 1.0)) * 2 + 1)
        general_interpolate_bilinear2d_aa_kernel[grid](
            output,
            input,
            N,
            C,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
            MAX_INTERPOLATE_H=max_interpolate_h,
            MAX_INTERPOLATE_W=max_interpolate_w,
            BLOCK_X=BLOCK_X,
            BLOCK_Y=BLOCK_Y,
        )
    else:
        upsample_bilinear2d_aa_kernel[grid](
            output,
            input,
            N,
            C,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
            BLOCK_X=BLOCK_X,
            BLOCK_Y=BLOCK_Y,
        )
    return output
