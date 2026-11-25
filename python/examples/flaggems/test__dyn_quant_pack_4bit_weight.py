import torch

from ._dyn_quant_pack_4bit_weight import _dyn_quant_pack_4bit_weight


def test__dyn_quant_pack_4bit_weight():
    weights = torch.arange(8, dtype=torch.uint8).view(2, 4)
    packed = _dyn_quant_pack_4bit_weight(
        weights, torch.ones(2, 1), torch.zeros(2), 4, 4, 2
    )
    assert packed.ndim == 1 and packed.numel() == 8 + 2 + 2
