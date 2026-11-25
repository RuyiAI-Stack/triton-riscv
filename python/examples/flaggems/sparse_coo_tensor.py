import torch

from .detach_copy import _copy_values


def sparse_coo_tensor(
    indices, values, size=None, *, dtype=None, device=None, is_coalesced=None
):
    copied_i = _copy_values(indices.to(torch.int64))
    copied_v = _copy_values(values if dtype is None else values.to(dtype))
    kwargs = {"dtype": copied_v.dtype, "device": device or copied_v.device}
    if is_coalesced is not None:
        kwargs["is_coalesced"] = is_coalesced
    if size is None:
        return torch.sparse_coo_tensor(copied_i, copied_v, **kwargs)
    return torch.sparse_coo_tensor(copied_i, copied_v, size, **kwargs)
