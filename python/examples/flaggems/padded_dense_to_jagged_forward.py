import torch

from .detach_copy import _copy_values


def padded_dense_to_jagged_forward(dense, offsets):
    pieces = []
    for row in range(offsets.numel() - 1):
        start = int(offsets[row])
        end = int(offsets[row + 1])
        pieces.append(dense[row, : end - start])
    return _copy_values(torch.cat(pieces))
