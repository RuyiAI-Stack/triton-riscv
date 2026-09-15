import torch

from .unique_dim import unique_dim


def test_unique_dim_values_inverse_counts():
    x = torch.tensor([[1, 2], [1, 2], [3, 4]], dtype=torch.int64, device="cpu")

    out = unique_dim(x, dim=0, sorted=True, return_inverse=True, return_counts=True)
    ref = torch.unique(x, dim=0, sorted=True, return_inverse=True, return_counts=True)

    for actual, expected in zip(out, ref):
        torch.testing.assert_close(actual, expected)
