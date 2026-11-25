import torch

from .detach_copy import _copy_values


def sparse_bsc_tensor_unsafe(ccol_indices, row_indices, values, size):
    return torch.sparse_csc_tensor(
        _copy_values(ccol_indices.to(torch.int64)),
        _copy_values(row_indices.to(torch.int64)),
        _copy_values(values),
        size,
        device=values.device,
    )
