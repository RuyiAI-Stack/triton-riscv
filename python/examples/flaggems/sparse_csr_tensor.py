import torch

from .detach_copy import _copy_values


def sparse_csr_tensor(
    crow_indices, col_indices, values, size, *, dtype=None, device=None
):
    return torch.sparse_csr_tensor(
        _copy_values(crow_indices.to(torch.int64)),
        _copy_values(col_indices.to(torch.int64)),
        _copy_values(values if dtype is None else values.to(dtype)),
        size,
        device=device or values.device,
    )
