import torch

from .thnn_differentiable_lstm_cell_backward import (
    thnn_differentiable_lstm_cell_backward,
)


def test_thnn_differentiable_lstm_cell_backward():
    grad_h = torch.randn(2, 4)
    grad_c = torch.randn(2, 4)
    gates = torch.randn(2, 16)
    state = torch.randn(2, 4)
    outs = thnn_differentiable_lstm_cell_backward(
        grad_h, grad_c, gates, gates, state, state
    )
    assert len(outs) == 6
    torch.testing.assert_close(outs[0], grad_h, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(outs[1], grad_c, rtol=1e-4, atol=1e-4)
