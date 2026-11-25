import torch

from ._sparse_log_softmax_backward_data import _sparse_log_softmax_backward_data


def test__sparse_log_softmax_backward_data():
    output = torch.log_softmax(torch.tensor([[1.0, 2.0], [0.0, 3.0]]), dim=1)
    grad = torch.ones_like(output)
    back = _sparse_log_softmax_backward_data(grad, output, 1, output.shape)
    assert back.shape == output.shape and torch.isfinite(back.to_dense()).all()
