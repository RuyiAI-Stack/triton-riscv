from .detach_copy import _copy_values


def _nested_tensor_from_mask(tensor, mask):
    return _copy_values(tensor[mask])
