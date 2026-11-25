import torch


def _jagged_to_padded_dense_forward(values, offsets, max_length, padding_value=0.0):
    batch = offsets.numel() - 1
    hidden = values.shape[-1] if values.ndim > 1 else 1
    out = torch.full(
        (batch, max_length, hidden),
        padding_value,
        dtype=values.dtype,
        device=values.device,
    )
    flat = values.reshape(-1, hidden)
    for row in range(batch):
        start = int(offsets[row])
        end = int(offsets[row + 1])
        out[row, : end - start] = flat[start:end]
    return out.squeeze(-1) if values.ndim == 1 else out
