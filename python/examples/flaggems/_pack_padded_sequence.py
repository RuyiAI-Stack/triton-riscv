import torch

from .detach_copy import _copy_values


def _pack_padded_sequence(input, lengths, batch_first=False):
    seq = input.transpose(0, 1) if batch_first else input
    packed = []
    for t in range(seq.shape[0]):
        alive = lengths > t
        packed.append(seq[t, alive])
    data = torch.cat(packed)
    batch_sizes = torch.tensor(
        [(lengths > t).sum() for t in range(seq.shape[0])], dtype=torch.int64
    )
    return _copy_values(data), batch_sizes
