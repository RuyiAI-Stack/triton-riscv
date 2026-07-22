import torch
import triton
import triton.language as tl


@triton.jit
def _jagged_to_padded_dense_forward_kernel(
    values,
    offsets,
    output,
    padding_value: tl.constexpr,
    batch_size: tl.constexpr,
    max_length: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for converting jagged tensor to padded dense tensor.

    Args:
        values: 1D tensor containing concatenated variable-length sequences
        offsets: 1D tensor of start positions for each sequence
        output: 2D output tensor of shape (batch_size, max_length)
        padding_value: scalar value for padding
        batch_size: number of sequences
        max_length: maximum length of each sequence
    """
    pid = tl.program_id(axis=0)
    batch_idx = pid

    if batch_idx >= batch_size:
        return

    # Get the start and end offset for this sequence
    seq_start = tl.load(offsets + batch_idx)
    seq_end = tl.load(offsets + batch_idx + 1)

    # Calculate the actual sequence length
    seq_length = seq_end - seq_start
    copy_length = tl.minimum(seq_length, max_length)

    # Compute the row offset in the output
    row_offset = batch_idx * max_length

    # Fill with padding value (vectorized per block)
    for j in tl.range(0, max_length, BLOCK_SIZE):
        out_offsets = row_offset + j + tl.arange(0, BLOCK_SIZE)
        out_mask = (j + tl.arange(0, BLOCK_SIZE)) < max_length
        tl.store(output + out_offsets, padding_value, mask=out_mask)

    # Copy actual values (vectorized per block)
    for j in tl.range(0, copy_length, BLOCK_SIZE):
        offsets_vec = seq_start + j + tl.arange(0, BLOCK_SIZE)
        mask = offsets_vec < seq_end

        values_vec = tl.load(values + offsets_vec, mask=mask, other=padding_value)

        out_offsets = row_offset + j + tl.arange(0, BLOCK_SIZE)
        out_mask = (j + tl.arange(0, BLOCK_SIZE)) < copy_length

        tl.store(output + out_offsets, values_vec, mask=out_mask)


def _jagged_to_padded_dense_forward(values, offsets, max_lengths, padding_value=0.0):
    """Convert a jagged (variable-length) tensor to a padded dense tensor.

    Args:
        values: 1D tensor containing concatenated variable-length sequences
        offsets: List of 1D tensors containing start positions for each sequence
        max_lengths: List of integers specifying maximum length for each batch dimension
        padding_value: Value to use for padding (default: 0.0)

    Returns:
        Padded dense tensor
    """
    # Currently only supports single batch dimension
    if not isinstance(offsets, (list, tuple)):
        offsets = [offsets]
    if not isinstance(max_lengths, (list, tuple)):
        max_lengths = [max_lengths]

    num_batch_dims = len(offsets)
    assert num_batch_dims == 1, (
        f"Only single batch dimension is supported, got {num_batch_dims}"
    )

    # Single batch dimension: 1D values, 1D offsets
    offsets_0 = offsets[0]
    batch_size = offsets_0.numel() - 1
    max_length = max_lengths[0]

    # Compute output shape
    # For single batch dim: (batch_size, max_length)
    output_shape = (batch_size, max_length)
    output = torch.empty(output_shape, dtype=values.dtype, device=values.device)

    def grid(meta):
        return (batch_size,)

    _jagged_to_padded_dense_forward_kernel[grid](
        values,
        offsets_0,
        output,
        padding_value,
        batch_size,
        max_length,
        # BLOCK_SIZE=128 balances occupancy and memory efficiency for typical sequence lengths
        BLOCK_SIZE=128,
    )

    return output
