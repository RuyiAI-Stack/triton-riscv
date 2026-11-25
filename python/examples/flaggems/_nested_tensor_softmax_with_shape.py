import torch


def _nested_tensor_softmax_with_shape(tensor, shape):
    _ = shape
    return torch.softmax(tensor, dim=-1)
