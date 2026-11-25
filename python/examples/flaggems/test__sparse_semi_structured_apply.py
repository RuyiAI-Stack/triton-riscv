import torch

from ._sparse_semi_structured_apply import _sparse_semi_structured_apply


def test__sparse_semi_structured_apply():
    torch.manual_seed(0)
    dense = torch.randn(2, 8)
    pruned = _sparse_semi_structured_apply(dense)
    assert pruned.shape == dense.shape
