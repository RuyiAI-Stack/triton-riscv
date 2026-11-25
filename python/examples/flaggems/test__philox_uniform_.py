import torch

from ._philox_uniform_ import _philox_uniform_


def test__philox_uniform_():
    tensor = torch.empty(8)
    _philox_uniform_(tensor)
    assert torch.all((tensor > 0) & (tensor < 1))
