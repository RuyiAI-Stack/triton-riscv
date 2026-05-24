import torch

from .copy import copy_


def contiguous(inp, memory_format=torch.contiguous_format):
    assert memory_format == torch.contiguous_format
    if inp.is_contiguous(memory_format=memory_format):
        return inp
    out = torch.empty_like(inp, memory_format=memory_format)
    return copy_(out, inp)
