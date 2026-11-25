import torch

from .detach_copy import _copy_values


def fft_ifftshift(self, dim=None):
    if dim is None:
        dim = tuple(range(self.ndim))
    if isinstance(dim, int):
        dim = (dim,)
    out = self
    for axis in dim:
        shift = -(out.shape[axis] // 2)
        out = torch.roll(out, shift, dims=axis)
    return _copy_values(out)
