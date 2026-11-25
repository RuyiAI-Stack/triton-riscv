"""One-dimensional CPU Triton histograms with uniform or explicit bins."""

from ._histogram_cpu import histogram_edges, uniform_edges


def histogram_bin_ct(self, bins=100, *, range=None, weight=None, density=False):
    x = self.reshape(-1, 1)
    edges = uniform_edges(x, [bins], range)
    return histogram_edges(x, edges, weight, density), edges[0]


def histogram_bins_tensor(self, bins, *, weight=None, density=False):
    return histogram_edges(self.reshape(-1, 1), [bins], weight, density), bins


def histogram(self, bins=100, *, range=None, weight=None, density=False):
    if isinstance(bins, int):
        return histogram_bin_ct(self, bins, range=range, weight=weight, density=density)
    return histogram_bins_tensor(self, bins, weight=weight, density=density)
