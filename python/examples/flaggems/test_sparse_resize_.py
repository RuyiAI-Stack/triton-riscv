import torch

from .sparse_resize_ import sparse_resize_


def test_sparse_resize_():
    tensor = torch.tensor([[1.0, 0.0], [0.0, 0.0]]).to_sparse()
    sparse_resize_(tensor, (2, 3), 2, 0)
    assert tuple(tensor.shape) == (2, 3)
