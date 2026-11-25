"""Multidimensional CPU Triton histogram, including weighted density."""

import torch
from ._histogram_cpu import histogram_edges, uniform_edges


def histogramdd(input, bins=10, range=None, weight=None, density=False):
    edges = (
        bins
        if isinstance(bins, (list, tuple))
        and all(isinstance(b, torch.Tensor) for b in bins)
        else uniform_edges(input, bins, range)
    )
    return histogram_edges(input, edges, weight, density), tuple(edges)
