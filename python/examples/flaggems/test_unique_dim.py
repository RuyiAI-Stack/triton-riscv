import torch

from .unique_dim import unique_dim


def test_unique_dim():
    rows = torch.tensor([[1.0, 0.0], [1.0, 0.0], [2.0, 3.0]])
    values, inverse, counts = unique_dim(rows, 0, True, True, True)
    ref = torch.unique(
        rows, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    torch.testing.assert_close(values, ref[0], rtol=1e-4, atol=1e-4)
    assert torch.equal(inverse, ref[1]) and torch.equal(counts, ref[2])
