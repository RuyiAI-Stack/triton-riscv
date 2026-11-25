import torch

from .detach_copy import _copy_values


def nested_to_padded_tensor(self, padding=0.0, output_size=None):
    if output_size is None:
        output_size = self.shape
    out = torch.full(output_size, padding, dtype=self.dtype, device=self.device)
    slices = tuple(slice(0, dim) for dim in self.shape)
    out[slices] = self
    return _copy_values(out)
