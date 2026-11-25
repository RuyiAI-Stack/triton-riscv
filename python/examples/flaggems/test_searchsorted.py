import torch

from .searchsorted import searchsorted


def test_searchsorted():
    boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0])
    values = torch.tensor([0.0, 3.0, 6.0, 8.0])
    assert torch.equal(
        searchsorted(boundaries, values), torch.searchsorted(boundaries, values)
    )
