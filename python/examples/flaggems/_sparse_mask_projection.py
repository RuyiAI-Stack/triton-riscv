from .sparse_mask import sparse_mask


def _sparse_mask_projection(self, mask, before_shape):
    _ = before_shape
    return sparse_mask(self, mask)
