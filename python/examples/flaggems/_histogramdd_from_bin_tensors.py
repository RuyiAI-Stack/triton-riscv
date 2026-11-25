"""Histogram from explicit edges without runtime tuple pointer arguments."""

from ._histogram_cpu import histogram_edges


def _histogramdd_from_bin_tensors(self, bins, *, weight=None, density=False):
    return histogram_edges(self, bins, weight, density)


def _histogramdd_from_bin_tensors_out(self, bins, *, weight=None, density=False, out):
    result = histogram_edges(self, bins, weight, density)
    out.resize_(result.shape).copy_(result)
    return out
