import torch

from .cat import cat


def hstack(
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> torch.Tensor:
    if len(tensors) == 0:
        raise RuntimeError("hstack expected a non-empty TensorList")

    dim = 0 if tensors[0].dim() <= 1 else 1
    return cat(
        [tensor.view(1) if tensor.dim() == 0 else tensor for tensor in tensors],
        dim=dim,
    )
