from .scatter import scatter_


def scatter_add_(x, dim, index, src):
    return scatter_(x, dim, index, src, reduce="add")
