def _thnn_fused_lstm_cell_backward(grad_hy, grad_cy, cx, cy, workspace, has_bias=True):
    _ = (cx, workspace, has_bias)
    return grad_hy, grad_cy, grad_hy, grad_hy, grad_hy, grad_hy
