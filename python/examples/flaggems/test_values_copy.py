import torch

from .values_copy import values_copy


def test_values_copy():
    tensor = torch.eye(2).to_sparse()
    torch.testing.assert_close(
        values_copy(tensor), tensor.values(), rtol=1e-4, atol=1e-4
    )
