import torch

from .bucketize import bucketize


def test_bucketize():
    boundaries = torch.tensor([1.0, 2.0, 4.0])
    values = torch.tensor([0.0, 2.0, 3.0, 5.0])
    assert torch.equal(
        bucketize(values, boundaries), torch.bucketize(values, boundaries)
    )
