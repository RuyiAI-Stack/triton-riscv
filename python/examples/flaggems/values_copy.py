from .indices_copy import _sparse_indices


def values_copy(self):
    return _sparse_indices(self, "values")
