import torch

from .detach_copy import _copy_values


def sparse_csc_tensor(
    ccol_indices, row_indices, values, size, *, dtype=None, device=None
):
    return torch.sparse_csc_tensor(
        _copy_values(ccol_indices.to(torch.int64)),
        _copy_values(row_indices.to(torch.int64)),
        _copy_values(values if dtype is None else values.to(dtype)),
        size,
        device=device or values.device,
    )
