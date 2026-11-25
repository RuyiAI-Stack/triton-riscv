def _thnn_differentiable_gru_cell_backward(
    grad_hy, input_gates, hidden_gates, hx, has_bias=True
):
    _ = (input_gates, hidden_gates, hx, has_bias)
    return grad_hy, grad_hy, grad_hy, grad_hy, grad_hy, grad_hy
