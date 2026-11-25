def argsort(inp, dim=-1, descending=False):
    from .sort import sort

    _values, indices = sort(inp, dim=dim, descending=descending)
    return indices
