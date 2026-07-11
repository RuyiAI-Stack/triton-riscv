import torch

from .cat import cat


def hstack(
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> torch.Tensor:
    if len(tensors) == 0:
        raise RuntimeError("hstack expected a non-empty TensorList")

    normalized = [tensor.reshape(1) if tensor.ndim == 0 else tensor for tensor in tensors]
    dim = 0 if normalized[0].ndim == 1 else 1
    return cat(normalized, dim=dim)
