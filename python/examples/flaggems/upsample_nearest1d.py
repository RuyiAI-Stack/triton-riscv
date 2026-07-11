import torch
import triton
import triton.language as tl


@triton.jit
def upsample_nearest1d_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OL,
    IL,
    reciprocal_scale_l,
    BLOCK_SIZE: tl.constexpr,
    SAME_L: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * OL
    mask = idx < total
    ol = idx % OL
    nc = idx // OL

    if SAME_L:
        il = ol
    else:
        il = tl.minimum(
            tl.math.floor(ol.to(tl.float32) * reciprocal_scale_l).to(tl.int32),
            IL - 1,
        )

    offset_i = nc * IL + il
    data = tl.load(ptr_i + offset_i, mask=mask)
    tl.store(ptr_o + idx, data, mask=mask)


def upsample_nearest1d(
    input: torch.Tensor,
    output_size: tuple[int] | None = None,
    scales: float | None = None,
) -> torch.Tensor:
    assert input.ndim == 3, "The ndim of input must be 3"
    assert output_size is not None or scales is not None, (
        "Either output_size or scales should be defined."
    )

    OL = (
        output_size[0]
        if output_size is not None
        else int(input.shape[2] * scales)
    )
    N, C, IL = input.shape

    if scales is not None:
        reciprocal_scale_l = float(
            torch.tensor(1.0 / scales, dtype=torch.float32).item()
        )
    else:
        # Use float32 division to match PyTorch's behavior
        reciprocal_scale_l = float(
            (
                torch.tensor(IL, dtype=torch.float32)
                / torch.tensor(OL, dtype=torch.float32)
            ).item()
        )

    # allocate output
    output = torch.empty((N, C, OL), device=input.device, dtype=input.dtype)
    total_threads = N * C * OL
    grid = (triton.cdiv(total_threads, 1024),)
    same_l = IL == OL

    upsample_nearest1d_kernel[grid](
        output,
        input,
        N,
        C,
        OL,
        IL,
        reciprocal_scale_l,
        BLOCK_SIZE=1024,
        SAME_L=same_l,
        USE_INT32_IDX=OL * N * C < 2**31,
    )
    return output
