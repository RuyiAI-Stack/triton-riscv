import torch

from .detach_copy import _copy_values


def _dyn_quant_pack_4bit_weight(
    weights, scales_zeros, bias, block_size, in_features, out_features
):
    _ = block_size
    if in_features <= 0 or in_features % 2:
        raise RuntimeError("in_features must be positive and even")
    pieces = [_copy_values(weights).reshape(-1).to(torch.float32)]
    pieces.append(_copy_values(scales_zeros).reshape(-1).to(torch.float32))
    if bias is not None:
        pieces.append(_copy_values(bias).reshape(-1).to(torch.float32))
    _ = (out_features, in_features)
    return torch.cat(pieces)


__all__ = ["_dyn_quant_pack_4bit_weight"]
