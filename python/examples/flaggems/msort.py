def msort(inp):
    from .sort import sort

    values, _indices = sort(inp, dim=0, descending=False)
    return values
