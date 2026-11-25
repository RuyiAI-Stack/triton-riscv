import torch

from ._thnn_fused_lstm_cell_backward import _thnn_fused_lstm_cell_backward


def test__thnn_fused_lstm_cell_backward():
    grad_h = torch.randn(2, 4)
    grad_c = torch.randn(2, 4)
    state = torch.randn(2, 4)
    outs = _thnn_fused_lstm_cell_backward(grad_h, grad_c, state, state, state)
    assert len(outs) == 6
    torch.testing.assert_close(outs[0], grad_h, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(outs[1], grad_c, rtol=1e-4, atol=1e-4)
