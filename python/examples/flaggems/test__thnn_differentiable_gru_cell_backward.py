import torch

from ._thnn_differentiable_gru_cell_backward import (
    _thnn_differentiable_gru_cell_backward,
)


def test__thnn_differentiable_gru_cell_backward():
    grad = torch.randn(2, 4)
    gates = torch.randn(2, 12)
    hx = torch.randn(2, 4)
    outs = _thnn_differentiable_gru_cell_backward(grad, gates, gates, hx)
    assert len(outs) == 6
    torch.testing.assert_close(outs[0], grad, rtol=1e-4, atol=1e-4)
