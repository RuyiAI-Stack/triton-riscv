import torch

from ._sparse_semi_structured_tile import _sparse_semi_structured_tile


def test__sparse_semi_structured_tile():
    torch.manual_seed(0)
    dense = torch.randn(2, 8)
    values, mask = _sparse_semi_structured_tile(dense)
    assert values.shape == dense.shape and mask.shape == dense.shape
    assert int(mask.view(2, -1, 4).sum(-1).max()) <= 2
