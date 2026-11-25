import torch

from .detach_copy import _copy_values


def _nested_select_backward(grad, indices, grad_shape):
    out = torch.zeros(grad_shape, dtype=grad.dtype, device=grad.device)
    out[indices] = grad
    return _copy_values(out)
