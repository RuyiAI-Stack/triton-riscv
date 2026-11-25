import torch

from ._sparse_softmax import _sparse_softmax


def test__sparse_softmax():
    tensor = torch.tensor([[1.0, 2.0], [0.0, 3.0]]).to_sparse()
    got = _sparse_softmax(tensor, dim=1)
    torch.testing.assert_close(
        got.to_dense(), torch.softmax(tensor.to_dense(), dim=1), rtol=1e-4, atol=1e-4
    )
