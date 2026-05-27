import torch

from .neg import neg


def resolve_neg(A: torch.Tensor):
    return neg(A) if A.is_neg() else A
