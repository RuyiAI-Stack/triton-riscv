import torch

from .ne import ne, ne_scalar


def not_equal(A, B):
    if isinstance(A, torch.Tensor):
        return ne(A, B)
    if isinstance(B, torch.Tensor):
        return ne_scalar(B, A)
    return torch.tensor(A != B)


def not_equal_scalar(A, B):
    if isinstance(A, torch.Tensor):
        return ne_scalar(A, B)
    return not_equal(A, B)
