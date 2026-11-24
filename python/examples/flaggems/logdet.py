import torch

from .linalg_slogdet import linalg_slogdet


def logdet(A):
    sign, logabs = linalg_slogdet(A)
    return torch.where(sign > 0, logabs, torch.full_like(logabs, float("nan")))
