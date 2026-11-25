def _sparse_softmax_backward_data(grad, output, dim, input_shape):
    _ = input_shape
    grad_d = grad.to_dense() if grad.is_sparse else grad
    out_d = output.to_dense() if output.is_sparse else output
    dotted = (grad_d * out_d).sum(dim=dim, keepdim=True)
    return ((grad_d - dotted) * out_d).to_sparse()
