import torch
import triton
import triton.language as tl


@triton.jit
def histc_kernel(
    inp_ptr,
    out_ptr,
    n_elements,
    bins: tl.constexpr,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute histogram of input tensor.
    Each thread processes BLOCK_SIZE elements, computing which bin they belong to
    and atomically incrementing the corresponding bin counter.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load input values
    inp_val = tl.load(inp_ptr + offset, mask=mask, other=0.0)

    # Convert to float32 for computation
    inp_val = inp_val.to(tl.float32)

    # Compute bin range
    bin_width = (max_val - min_val) / bins

    # Compute bin indices
    # Elements equal to max_val go to the last bin (bins - 1)
    # Elements outside [min_val, max_val] or NaN are ignored
    bin_idx = ((inp_val - min_val) / bin_width).to(tl.int32)

    # Clamp to valid range [0, bins-1] for elements in range
    # Elements outside range or NaN should be excluded
    in_range = (inp_val >= min_val) & (inp_val <= max_val)

    # Handle edge case: elements exactly equal to max go to last bin
    bin_idx = tl.where(inp_val == max_val, bins - 1, bin_idx)
    bin_idx = tl.where(bin_idx < 0, 0, bin_idx)
    bin_idx = tl.where(bin_idx >= bins, bins - 1, bin_idx)

    # Only count elements in range (excludes NaN via the comparison)
    valid_mask = mask & in_range

    # Atomic add to histogram bins
    # We need to iterate through each element and add to the appropriate bin
    for i in range(BLOCK_SIZE):
        if tl.load(valid_mask.to(tl.int8).reshape(BLOCK_SIZE) + i) != 0:
            idx = tl.load(bin_idx.reshape(BLOCK_SIZE) + i)
            tl.atomic_add(out_ptr + idx, 1.0, sem="relaxed")


@triton.jit
def histc_kernel_simple(
    inp_ptr,
    out_ptr,
    n_elements,
    bins,
    min_val,
    max_val,
):
    """Count one output bin per program without shared atomic updates."""
    target_bin = tl.program_id(0)
    count = tl.full((), 0, dtype=tl.int32)

    for offset in range(0, n_elements):
        inp_val = tl.load(inp_ptr + offset).to(tl.float32)
        in_range = (inp_val >= min_val) & (inp_val <= max_val)
        normalized = (inp_val - min_val) * bins / (max_val - min_val)
        safe_normalized = tl.where(in_range, normalized, 0.0)
        bin_idx = tl.floor(safe_normalized).to(tl.int32)
        bin_idx = tl.where(inp_val == max_val, bins - 1, bin_idx)
        count += (in_range & (bin_idx == target_bin)).to(tl.int32)

    tl.store(out_ptr + target_bin, count.to(out_ptr.type.element_ty))


def histc(inp, bins=100, min=0, max=0):
    """Compute the histogram of a tensor.

    Args:
        inp: Input tensor
        bins: Number of histogram bins (default: 100)
        min: Lower end of the range (inclusive). If min == max == 0, uses data min.
        max: Upper end of the range (inclusive). If min == max == 0, uses data max.

    Returns:
        Tensor: Histogram represented as a tensor of shape (bins,)
    """
    # Ensure input is contiguous
    inp = inp.contiguous()

    # Get min and max values
    min_val = float(min)
    max_val = float(max)

    if min_val == 0 and max_val == 0:
        # Use actual min/max of the data
        min_val = float(inp.min().item())
        max_val = float(inp.max().item())

    # Handle edge case where min == max
    if min_val == max_val:
        # All elements go to the first bin if they equal min_val
        out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)
        # Count how many elements equal min_val (excluding NaN)
        count = ((inp == min_val) & ~torch.isnan(inp)).sum().item()
        out[0] = count
        return out

    # Create output histogram tensor
    out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)

    n_elements = inp.numel()

    if n_elements == 0:
        return out

    grid = (bins,)

    histc_kernel_simple[grid](
        inp,
        out,
        n_elements,
        bins,
        min_val,
        max_val,
    )

    return out
