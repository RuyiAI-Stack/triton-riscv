from .nonzero import nonzero


def nonzero_numpy(inp):
    out = nonzero(inp, as_tuple=False)
    return list(out.unbind(dim=1))
