import torch

from ._sparse_log_softmax import _sparse_log_softmax


def test__sparse_log_softmax():
    tensor = torch.tensor([[1.0, 2.0], [0.0, 3.0]]).to_sparse()
    got = _sparse_log_softmax(tensor, dim=1)
    torch.testing.assert_close(
        got.to_dense(),
        torch.log_softmax(tensor.to_dense(), dim=1),
        rtol=1e-4,
        atol=1e-4,
    )
