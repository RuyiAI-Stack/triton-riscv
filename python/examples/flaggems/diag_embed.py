import torch

from .copy import copy_


def diag_embed(x, offset=0, dim1=-2, dim2=-1):
    rank = x.ndim + 1

    assert dim1 >= -rank and dim1 < rank, f"Invalid dim1: {dim1}"
    assert dim2 >= -rank and dim2 < rank, f"Invalid dim2: {dim2}"
    dim1 = dim1 % rank
    dim2 = dim2 % rank

    assert dim1 != dim2, "diagonal dimensions cannot be identical"

    if dim1 > dim2:
        offset = -offset
        dim1, dim2 = dim2, dim1

    last_dim = x.size(-1) + abs(offset)

    y_shape = list(x.shape)
    y_shape.pop()
    y_shape.insert(dim1, last_dim)
    y_shape.insert(dim2, last_dim)

    y = torch.zeros(y_shape, dtype=x.dtype, device=x.device)
    y_diagonal_view = torch.diagonal(y, offset, dim1, dim2)
    copy_(y_diagonal_view, x)

    return y
