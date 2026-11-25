from .detach_copy import _copy_values
from .sparse_coo_tensor import sparse_coo_tensor


def sparse_mask(self, mask):
    dense = self.to_dense() if self.is_sparse else self
    masked = dense.sparse_mask(mask)
    return sparse_coo_tensor(
        masked.indices(), _copy_values(masked.values()), masked.shape
    )
