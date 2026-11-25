import torch

from .triangular_indices import triangular_indices


def test_triangular_indices():
    assert torch.equal(triangular_indices(4, 3, -1), torch.tril_indices(4, 3, -1))
