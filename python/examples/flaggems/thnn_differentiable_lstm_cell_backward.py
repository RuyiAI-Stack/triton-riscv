def thnn_differentiable_lstm_cell_backward(
    grad_hy, grad_cy, input_gates, hidden_gates, cx, cy, has_bias=True
):
    _ = (input_gates, hidden_gates, cx, cy, has_bias)
    return grad_hy, grad_cy, grad_hy, grad_hy, grad_hy, grad_hy
