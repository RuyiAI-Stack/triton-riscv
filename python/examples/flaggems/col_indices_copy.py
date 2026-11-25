from .indices_copy import _sparse_indices


def col_indices_copy(self):
    return _sparse_indices(self, "col")
