import torch

from .detach_copy import _copy_values


def nonzero_static(input, size=None):
    mask = input != 0
    idx = mask.nonzero(as_tuple=False)
    if size is None:
        return _copy_values(idx)
    out = torch.zeros(size, input.ndim, dtype=torch.int64, device=input.device)
    n = min(size, idx.shape[0])
    if n:
        out[:n] = idx[:n]
    return _copy_values(out)
