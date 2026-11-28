import torch
import triton
import triton.language as tl

# channel_shuffle: aten::channel_shuffle as a value copy. Input is
# (*, C, H, W) with C divisible by groups; the kernel decodes each output
# element into (n, c, h, w) and gathers from the group-transposed channel
# (new channel = channel_in_group * groups + group_id).


@triton.jit
def channel_shuffle_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    N,
    C,
    H,
    W,
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Strides for contiguous NCHW
    sN = C * H * W
    sC = H * W
    sH = W
    # Decode output linear index into (n, c, h, w)
    n = offsets // sN
    rem1 = offsets - n * sN
    c = rem1 // sC
    rem2 = rem1 - c * sC
    h = rem2 // sH
    w = rem2 - h * sH
    # Channel shuffle: (n, groups, C//groups, h, w) -> (n, C//groups, groups, h, w)
    c_per_group = C // groups
    group_id = c // c_per_group
    channel_in_group = c - group_id * c_per_group
    # offsets enumerate the INPUT; the shuffled channel position receives
    # the input value (scatter direction, as upstream).
    new_c = channel_in_group * groups + group_id
    val = tl.load(in_ptr + offsets, mask=mask)
    out_offset = n * sN + new_c * sC + h * sH + w
    tl.store(out_ptr + out_offset, val, mask=mask)


def channel_shuffle(input: torch.Tensor, groups: int) -> torch.Tensor:
    """Value copy of the group-transposed channel shuffle over (*, C, H, W)."""
    x = input
    if not x.is_contiguous():
        x = x.contiguous()
    if x.ndim < 3:
        raise ValueError(
            f"Input must have at least 3 dimensions (C, H, W), got {x.ndim}"
        )
    N, C, H, W = x.shape[-4:]
    g = int(groups)
    assert g > 0, "groups must be > 0"
    assert C % g == 0, f"C ({C}) must be divisible by groups ({g})"
    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    channel_shuffle_kernel[grid](
        x,
        out,
        n_elements,
        N,
        C,
        H,
        W,
        g,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
