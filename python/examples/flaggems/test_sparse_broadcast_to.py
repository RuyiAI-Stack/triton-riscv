import torch

from .sparse_broadcast_to import sparse_broadcast_to


def test_sparse_broadcast_to():
    tensor = torch.tensor([[1.0, 0.0]]).to_sparse()
    out = sparse_broadcast_to(tensor, (2, 2))
    assert tuple(out.shape) == (2, 2)
