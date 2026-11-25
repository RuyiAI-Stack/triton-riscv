from .indices_copy import _sparse_indices


def ccol_indices_copy(self):
    return _sparse_indices(self, "ccol")
