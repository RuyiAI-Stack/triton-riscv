import torch

from .detach_copy import _copy_values


def sparse_bsr_tensor_unsafe(crow_indices, col_indices, values, size):
    return torch.sparse_csr_tensor(
        _copy_values(crow_indices.to(torch.int64)),
        _copy_values(col_indices.to(torch.int64)),
        _copy_values(values),
        size,
        device=values.device,
    )
