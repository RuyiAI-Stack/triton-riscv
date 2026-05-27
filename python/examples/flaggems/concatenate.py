import torch

from .cat import cat


def concatenate(
    A: tuple[torch.Tensor, ...] | list[torch.Tensor], dim: int = 0
) -> torch.Tensor:

    return cat(A, dim=dim)
