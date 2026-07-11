import torch

from .repeat import repeat


def tile(inp: torch.Tensor, dims) -> torch.Tensor:
    dims = tuple(int(value) for value in dims)
    if len(dims) < inp.ndim:
        dims = (1,) * (inp.ndim - len(dims)) + dims
    return repeat(inp, dims)
