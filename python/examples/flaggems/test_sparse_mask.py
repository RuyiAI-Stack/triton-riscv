import torch

from .sparse_mask import sparse_mask


def test_sparse_mask():
    dense = torch.arange(4.0).view(2, 2)
    mask = torch.tensor([[0.0, 1.0], [0.0, 0.0]]).to_sparse()
    masked = sparse_mask(dense, mask)
    assert masked._nnz() == 1
