import torch

from .cat import cat


def stack(
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor], dim: int = 0
) -> torch.Tensor:
    if len(tensors) == 0:
        raise RuntimeError("stack expected a non-empty TensorList")

    ndim = tensors[0].ndim
    if dim < -ndim - 1 or dim > ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim - 1}, {ndim}], but got {dim})"
        )
    dim %= ndim + 1
    expected_shape = tensors[0].shape
    for index, tensor in enumerate(tensors[1:], 1):
        if tensor.shape != expected_shape:
            raise RuntimeError(
                "stack expects each tensor to be equal size, but got "
                f"{list(expected_shape)} at entry 0 and {list(tensor.shape)} "
                f"at entry {index}"
            )
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim).clone()
    return cat([tensor.unsqueeze(dim) for tensor in tensors], dim=dim)
