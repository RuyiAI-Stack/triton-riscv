from .cumsum import cumsum


def cumsum_(inp, dim, *, dtype=None):
    if dtype is not None and dtype != inp.dtype:
        raise RuntimeError(
            "in-place cumsum requires the input and output dtype to match"
        )
    inp.copy_(cumsum(inp, dim, dtype=inp.dtype))
    return inp
