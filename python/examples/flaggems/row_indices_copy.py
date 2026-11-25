from .indices_copy import _sparse_indices


def row_indices_copy(self):
    return _sparse_indices(self, "row")
