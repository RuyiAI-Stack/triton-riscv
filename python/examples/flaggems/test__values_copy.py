import torch

from ._values_copy import _values_copy


def test__values_copy():
    tensor = torch.eye(2).to_sparse()
    torch.testing.assert_close(
        _values_copy(tensor), tensor.values(), rtol=1e-4, atol=1e-4
    )
