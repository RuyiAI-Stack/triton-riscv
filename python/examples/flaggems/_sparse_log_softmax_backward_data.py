def _sparse_log_softmax_backward_data(grad, output, dim, input_shape):
    _ = input_shape
    grad_d = grad.to_dense() if grad.is_sparse else grad
    out_d = output.to_dense() if output.is_sparse else output
    return (grad_d - grad_d.sum(dim=dim, keepdim=True) * out_d.exp()).to_sparse()
