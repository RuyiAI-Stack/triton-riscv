from ._sparse_semi_structured_tile import _sparse_semi_structured_tile


def _sparse_semi_structured_apply(dense):
    values, mask = _sparse_semi_structured_tile(dense)
    return values
