import torch

from .copy import copy_


def diagonal_backward(
    grad_output,
    input_sizes,
    offset,
    dim1,
    dim2,
):
    grad_input = torch.zeros(
        input_sizes, dtype=grad_output.dtype, device=grad_output.device
    )
    diag = torch.diagonal(grad_input, offset, dim1, dim2)
    copy_(diag, grad_output)
    return grad_input
