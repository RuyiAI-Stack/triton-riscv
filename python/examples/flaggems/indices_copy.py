from .detach_copy import _copy_values


def _sparse_indices(tensor, kind):
    if kind == "values":
        return _copy_values(tensor._values() if tensor.is_sparse else tensor.values())
    if kind == "indices":
        return _copy_values(tensor._indices() if tensor.is_sparse else tensor.indices())
    if kind == "crow":
        return _copy_values(tensor.crow_indices())
    if kind == "col":
        return _copy_values(tensor.col_indices())
    if kind == "ccol":
        return _copy_values(tensor.ccol_indices())
    if kind == "row":
        return _copy_values(tensor.row_indices())
    raise KeyError(kind)


def indices_copy(self):
    return _sparse_indices(self, "indices")
