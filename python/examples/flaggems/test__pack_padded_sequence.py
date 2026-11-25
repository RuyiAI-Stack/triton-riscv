import torch

from ._pack_padded_sequence import _pack_padded_sequence


def test__pack_padded_sequence():
    seq = torch.arange(12.0).view(3, 2, 2)
    data, sizes = _pack_padded_sequence(seq, torch.tensor([3, 1]))
    assert data.shape[0] == 4 and int(sizes[0]) == 2
