from ._nested_tensor_from_mask import _nested_tensor_from_mask


def _nested_tensor_from_mask_left_aligned(tensor, mask):
    return _nested_tensor_from_mask(tensor, mask)
