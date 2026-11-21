import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _pad_packed_sequence_kernel(
    data,
    batch_sizes,
    time_offsets,
    output,
    padding_value,
    num_steps: tl.constexpr,
    max_batch: tl.constexpr,
    feature_size: tl.constexpr,
    out_stride_t,
    out_stride_b,
    NUM_TILES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter a packed sequence back into a padded (T, B, *) tensor.

    Each program handles one (t, b) output row: it either copies the
    corresponding packed feature vector or fills the row with padding_value.

    Args:
        data: 2D packed tensor of shape (sum(batch_sizes), feature_size)
        batch_sizes: 1D int64 tensor, number of active batch elements per step
        time_offsets: 1D int64 tensor, exclusive prefix sum of batch_sizes
        output: flattened padded output buffer
        padding_value: scalar fill value for padded positions
        num_steps: number of valid time steps (batch_sizes.numel())
        max_batch: batch size (batch_sizes[0])
        feature_size: trailing feature dimension of data
        out_stride_t, out_stride_b: output row strides for time / batch axes
        NUM_TILES: number of BLOCK_SIZE feature tiles (cdiv(feature_size, BLOCK_SIZE))
        BLOCK_SIZE: feature tile size
    """
    pid = tl.program_id(0)
    t = pid // max_batch
    b = pid % max_batch

    row_offset = t * out_stride_t + b * out_stride_b

    # Positions past the packed sequence length for this batch element are
    # padded; t >= num_steps is always padding. The active flag is a scalar
    # predicate (no data-dependent branches), so both paths are masked selects.
    step_active = t < num_steps
    bs_t = tl.load(batch_sizes + t, mask=step_active, other=0)
    row_active = step_active & (b < bs_t)

    src_row = tl.load(time_offsets + t, mask=row_active, other=0) + b
    src_offset = src_row * feature_size
    for j in tl.static_range(NUM_TILES):
        cols = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < feature_size
        vals = tl.load(data + src_offset + cols, mask=mask & row_active, other=0.0)
        out_vals = tl.where(row_active, vals, padding_value)
        tl.store(output + row_offset + cols, out_vals, mask=mask)


def _pad_packed_sequence(data, batch_sizes, batch_first, padding_value, total_length):
    """aten::_pad_packed_sequence: unpack a packed sequence into a padded tensor.

    Inverse of ``_pack_padded_sequence``: scatters ``data`` (shape
    ``(sum(batch_sizes), *)``) into a padded ``(T, B, *)`` tensor (or
    ``(B, T, *)`` when ``batch_first``) and returns it together with the
    per-batch-element sequence lengths.

    Args:
        data: packed values tensor.
        batch_sizes: 1D CPU int64 tensor of active counts per time step.
        batch_first: if True, output layout is (B, T, *) instead of (T, B, *).
        padding_value: value used for padded positions.
        total_length: pad the time dimension out to at least this many steps;
            values <= 0 mean use the number of steps in ``batch_sizes``.

    Returns:
        Tuple of (padded output tensor, lengths tensor).
    """
    logger.debug("GEMS _PAD_PACKED_SEQUENCE")

    num_steps = batch_sizes.numel()
    max_batch = int(batch_sizes[0].item())

    total_steps = num_steps
    if total_length > 0:
        total_steps = max(total_steps, int(total_length))

    # batch_sizes is a CPU int64 tensor (aten contract). Derive the exclusive
    # prefix sum (packed offset of each time step) and per-element lengths on
    # host, then move the small metadata tensors to the compute device.
    batch_sizes_cpu = batch_sizes.to(dtype=torch.int64, device="cpu")
    time_offsets_cpu = torch.zeros_like(batch_sizes_cpu)
    time_offsets_cpu[1:] = torch.cumsum(batch_sizes_cpu[:-1], dim=0)
    lengths = torch.searchsorted(
        torch.flip(batch_sizes_cpu, [0]),
        torch.arange(max_batch, dtype=torch.int64),
        right=True,
    )
    lengths = (num_steps - lengths).to(torch.int64)

    metadata_dev = torch.cat([batch_sizes_cpu, time_offsets_cpu]).to(data.device)
    batch_sizes_dev = metadata_dev[:num_steps]
    time_offsets = metadata_dev[num_steps:]

    feature_shape = data.shape[1:]
    feature_size = 1
    for s in feature_shape:
        feature_size *= s

    output = torch.empty(
        (total_steps, max_batch, *feature_shape),
        dtype=data.dtype,
        device=data.device,
    )
    out_flat = output.view(total_steps, max_batch, feature_size)

    out_stride_t = out_flat.stride(0)
    out_stride_b = out_flat.stride(1)

    grid = (total_steps * max_batch,)
    # BLOCK_SIZE=1024 balances occupancy and register use for typical feature
    # widths; larger features are handled by the unrolled tile loop.
    BLOCK_SIZE = min(1024, max(1, triton.next_power_of_2(feature_size)))
    num_tiles = triton.cdiv(feature_size, BLOCK_SIZE)

    _pad_packed_sequence_kernel[grid](
        data.reshape(-1),
        batch_sizes_dev,
        time_offsets,
        out_flat,
        padding_value,
        num_steps,
        max_batch,
        feature_size,
        out_stride_t,
        out_stride_b,
        NUM_TILES=num_tiles,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if batch_first:
        output = output.transpose(0, 1).contiguous()

    return output, lengths.cpu()
