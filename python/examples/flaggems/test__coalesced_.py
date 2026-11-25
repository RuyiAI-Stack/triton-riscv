import torch

from ._coalesced_ import _coalesced_


def test__coalesced_():
    tensor = torch.eye(2).to_sparse()
    assert _coalesced_(tensor) is True
