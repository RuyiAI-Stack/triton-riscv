import torch

from .thnn_fused_gru_cell_backward import thnn_fused_gru_cell_backward


def test_thnn_fused_gru_cell_backward():
    grad = torch.randn(2, 4)
    outs = thnn_fused_gru_cell_backward(grad, torch.randn(2, 12))
    assert len(outs) == 4
    torch.testing.assert_close(outs[0], grad, rtol=1e-4, atol=1e-4)
