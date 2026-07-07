import torch

from .cat import cat


def vstack(tensors: list):
    return cat(torch.atleast_2d(tensors), dim=0)
