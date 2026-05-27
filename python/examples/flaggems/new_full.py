import torch

from .fill import fill_scalar_, fill_tensor_
from .full import check_dtype


def new_full(
    self,
    size,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    if device is None:
        device = self.device
    if dtype is None:
        dtype = self.dtype
    fill_value = check_dtype(fill_value, dtype, device)
    out = torch.empty(size, device=device, dtype=dtype)
    if out.numel() == 0:
        return out
    if isinstance(fill_value, torch.Tensor):
        fill_tensor_(out, fill_value)
    else:
        fill_scalar_(out, fill_value)
    return out
