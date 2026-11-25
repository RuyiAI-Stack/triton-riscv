import torch

from .detach_copy import _copy_values
from .sparse_coo_tensor import sparse_coo_tensor


def coalesce(self):
    indices = self._indices()
    values = self._values()
    if indices.numel() == 0:
        return self
    keys = indices[0] * (self.shape[-1] if self.ndim > 1 else 1) + (
        indices[1] if indices.shape[0] > 1 else 0
    )
    order = torch.argsort(keys)
    indices = indices[:, order]
    values = _copy_values(values[order])
    unique_keys = []
    unique_values = []
    acc = values[0].clone()
    current = indices[:, 0].clone()
    for pos in range(1, values.shape[0]):
        if torch.equal(indices[:, pos], current):
            acc = acc + values[pos]
        else:
            unique_keys.append(current)
            unique_values.append(acc)
            current = indices[:, pos].clone()
            acc = values[pos].clone()
    unique_keys.append(current)
    unique_values.append(acc)
    new_indices = torch.stack(unique_keys, dim=1)
    new_values = torch.stack(unique_values)
    return sparse_coo_tensor(new_indices, new_values, self.shape, is_coalesced=True)
