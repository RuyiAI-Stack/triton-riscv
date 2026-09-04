from .sort import sort_stable


def argsort(inp, dim=-1, descending=False):
    _, indices = sort_stable(inp, stable=True, dim=dim, descending=descending)
    return indices
