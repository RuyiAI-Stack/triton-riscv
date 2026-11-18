import torch
import triton
import triton.language as tl


@triton.jit
def channel_shuffle_kernel(
    in_ptr,
    out_ptr,
    numel,
    channels,
    spatial_size,
    groups,
    channels_per_group,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < numel

    spatial = offsets % spatial_size
    channel_linear = offsets // spatial_size
    out_channel = channel_linear % channels
    batch_idx = channel_linear // channels

    group_idx = out_channel % groups
    channel_in_group = out_channel // groups
    in_channel = group_idx * channels_per_group + channel_in_group

    input_offsets = ((batch_idx * channels) + in_channel) * spatial_size + spatial
    values = tl.load(in_ptr + input_offsets, mask=active)
    tl.store(out_ptr + offsets, values, mask=active)


def channel_shuffle(input: torch.Tensor, groups: int) -> torch.Tensor:
    x = input.contiguous() if not input.is_contiguous() else input
    if x.ndim < 3:
        raise ValueError(
            f"Input must have at least 3 dimensions (N, C, *), got {x.ndim}"
        )

    channels = x.shape[1]
    groups = int(groups)
    assert groups > 0, "groups must be > 0"
    assert channels % groups == 0, (
        f"C ({channels}) must be divisible by groups ({groups})"
    )

    out = torch.empty_like(x)
    if out.numel() == 0:
        return out

    spatial_size = x.numel() // (x.shape[0] * channels)
    block_size = 256
    grid = (triton.cdiv(out.numel(), block_size),)

    channel_shuffle_kernel[grid](
        x,
        out,
        out.numel(),
        channels,
        spatial_size,
        groups,
        channels // groups,
        BLOCK_SIZE=block_size,
    )
    return out
