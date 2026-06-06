import torch
import triton
import triton.language as tl


@triton.jit
def upsample_nearest3d_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OD,
    OH,
    OW,
    ID,
    IH,
    IW,
    reciprocal_scale_d,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
    SAME_D: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        pid0 = tl.program_id(axis=0)
    else:
        pid0 = tl.program_id(axis=0).to(tl.int64)
    nc_stride = tl.num_programs(axis=1)
    NC = N * C
    nc_iter = tl.program_id(axis=1)

    idx = pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_spatial_size = OD * OH * OW

    mask = idx < total_spatial_size

    ow = idx % OW
    oh = (idx // OW) % OH
    od = idx // (OW * OH)

    if SAME_D:
        id = od
    else:
        id = tl.minimum(
            tl.math.floor(od.to(tl.float32) * reciprocal_scale_d).to(tl.int32),
            ID - 1,
        )

    if SAME_H:
        ih = oh
    else:
        ih = tl.minimum(
            tl.math.floor(oh.to(tl.float32) * reciprocal_scale_h).to(tl.int32),
            IH - 1,
        )

    if SAME_W:
        iw = ow
    else:
        iw = tl.minimum(
            tl.math.floor(ow.to(tl.float32) * reciprocal_scale_w).to(tl.int32),
            IW - 1,
        )

    offset_o = nc_iter * (OD * OH * OW) + idx
    offset_i = nc_iter * (ID * IH * IW) + (id * IH * IW + ih * IW + iw)

    src_nc_stride = nc_stride * (ID * IH * IW)
    dst_nc_stride = nc_stride * (OD * OH * OW)

    while nc_iter < NC:
        data = tl.load(ptr_i + offset_i, mask=mask)
        tl.store(ptr_o + offset_o, data, mask=mask)

        offset_i += src_nc_stride
        offset_o += dst_nc_stride
        nc_iter += nc_stride


def upsample_nearest3d(
    input: torch.Tensor,
    output_size: tuple[int, int, int],
    scales_d: float | None = None,
    scales_h: float | None = None,
    scales_w: float | None = None,
) -> torch.Tensor:
    assert input.ndim == 5, "The ndim of input must be 5"

    OD, OH, OW = output_size
    N, C, ID, IH, IW = input.shape

    def calculate_scale(in_sz, out_sz, s):
        if s is not None:
            return float(torch.tensor(1.0 / s, dtype=torch.float32).item())
        return float(
            (
                torch.tensor(in_sz, dtype=torch.float32)
                / torch.tensor(out_sz, dtype=torch.float32)
            ).item()
        )

    reciprocal_scale_d = calculate_scale(ID, OD, scales_d)
    reciprocal_scale_h = calculate_scale(IH, OH, scales_h)
    reciprocal_scale_w = calculate_scale(IW, OW, scales_w)

    output = torch.empty((N, C, OD, OH, OW), device=input.device, dtype=input.dtype)

    total_threads = OD * OH * OW
    grid = (
        triton.cdiv(total_threads, 1024),
        triton.cdiv(N * C, 4),
    )

    same_d = ID == OD
    same_h = IH == OH
    same_w = IW == OW

    upsample_nearest3d_kernel[grid](
        output,
        input,
        N,
        C,
        OD,
        OH,
        OW,
        ID,
        IH,
        IW,
        reciprocal_scale_d,
        reciprocal_scale_h,
        reciprocal_scale_w,
        BLOCK_SIZE=1024,
        SAME_D=same_d,
        SAME_H=same_h,
        SAME_W=same_w,
        USE_INT32_IDX=OD * OH * OW * N * C < 2**31,
    )
    return output
