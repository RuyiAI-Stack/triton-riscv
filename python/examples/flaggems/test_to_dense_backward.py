import torch

from .to_dense_backward import to_dense_backward


def test_to_dense_backward():
    base = torch.tensor([[1.0, 0.0], [0.0, 2.0]]).to_sparse()
    grad = torch.ones(2, 2)
    back = to_dense_backward(grad, base)
    assert back.is_sparse
