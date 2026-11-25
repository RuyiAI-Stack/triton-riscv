def thnn_fused_gru_cell_backward(grad_hy, workspace, has_bias=True):
    _ = (workspace, has_bias)
    return grad_hy, grad_hy, grad_hy, grad_hy
