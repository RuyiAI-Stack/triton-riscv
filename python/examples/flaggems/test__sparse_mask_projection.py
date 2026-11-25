import torch

from ._sparse_mask_projection import _sparse_mask_projection


def test__sparse_mask_projection():
    dense = torch.arange(4.0).view(2, 2)
    mask = torch.tensor([[0.0, 1.0], [0.0, 0.0]]).to_sparse()
    masked = _sparse_mask_projection(dense, mask, dense.shape)
    assert masked._nnz() == 1
