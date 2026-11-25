import torch

from .is_nonzero import is_nonzero


def test_is_nonzero():
    assert is_nonzero(torch.tensor([2.0])) is True
    assert is_nonzero(torch.tensor([0.0])) is False
