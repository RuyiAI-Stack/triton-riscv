from .indices_copy import _sparse_indices


def crow_indices(self):
    return _sparse_indices(self, "crow")
