from .detach_copy import _copy_values


def to_dense_backward(grad, input, masked_grad=False):
    _ = masked_grad
    if input.is_sparse:
        return grad.sparse_mask(input)
    return _copy_values(grad)
