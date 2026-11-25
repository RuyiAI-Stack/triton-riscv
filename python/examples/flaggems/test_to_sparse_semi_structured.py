import torch

from .to_sparse_semi_structured import to_sparse_semi_structured


def test_to_sparse_semi_structured():
    torch.manual_seed(0)
    dense = torch.randn(2, 8)
    values, mask = to_sparse_semi_structured(dense)
    assert tuple(values.shape) == tuple(dense.shape) and mask.shape == dense.shape
