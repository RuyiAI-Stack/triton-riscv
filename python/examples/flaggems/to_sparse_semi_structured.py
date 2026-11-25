from ._sparse_semi_structured_tile import _sparse_semi_structured_tile


def to_sparse_semi_structured(dense):
    values, mask = _sparse_semi_structured_tile(dense)
    return values.to_sparse(), mask
