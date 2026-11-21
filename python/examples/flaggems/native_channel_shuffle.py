import torch
import triton
import triton.language as tl

# native_channel_shuffle: aten::native_channel_shuffle as a value copy over
# (N, C, *spatial) with C divisible by groups. The kernel decodes each
# output row into (n, c_out) and gathers from the inverse-mapped input
# channel (c_in = (c_out % groups) * cpg + c_out // groups).


@triton.jit
def _native_channel_shuffle_kernel(
    x_ptr,
    o_ptr,
    numel,
    C,
    HW,
    cpg,
    groups,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    # Decode linear output index to (n, c_out, spatial)
    row = offs // HW
    hw = offs - row * HW
    n = row // C
    c_out = row - n * C
    # Channel shuffle: c_out = (c_in % cpg) * groups + (c_in // cpg)
    # Inverse: c_in = (c_out % groups) * cpg + (c_out // groups)
    within = c_out // groups
    group = c_out - within * groups
    c_in = group * cpg + within
    # Compute input linear index
    src = (n * C + c_in) * HW + hw
    val = tl.load(x_ptr + src, mask=mask)
    tl.store(o_ptr + offs, val, mask=mask)


def native_channel_shuffle(input: torch.Tensor, groups: int) -> torch.Tensor:
    """Value copy of the channel shuffle over (N, C, *spatial)."""
    x = input
    if not x.is_contiguous():
        x = x.contiguous()
    if x.ndim < 2:
        raise ValueError(
            f"Input must have at least 2 dimensions (N, C, ...), got {x.ndim}"
        )
    C = x.shape[1]
    HW = 1
    for d in x.shape[2:]:
        HW *= d
    g = int(groups)
    assert g > 0, "groups must be > 0"
    assert C % g == 0, f"C ({C}) must be divisible by groups ({g})"
    out = torch.empty_like(x)
    numel = x.numel()
    if numel == 0:
        return out
    cpg = C // g
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _native_channel_shuffle_kernel[grid](x, out, numel, C, HW, cpg, g, BLOCK=BLOCK)
    return out
