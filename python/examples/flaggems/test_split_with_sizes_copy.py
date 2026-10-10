from math import prod

import pytest
import torch

from .split_with_sizes_copy import split_with_sizes_copy


@pytest.mark.parametrize(
    "shape, split_sizes, dim",
    [((5, 4), [2, 0, 3], 0), ((6, 3), [1, 3, 2], -1)],
)
def test_split_with_sizes_copy_matches_torch(shape, split_sizes, dim):
    x = torch.arange(prod(shape), dtype=torch.float32, device="cpu")
    x = x.reshape(shape)
    if dim == -1:
        x = x.transpose(0, 1)

    out = split_with_sizes_copy(x, split_sizes, dim=dim)
    ref = torch.ops.aten.split_with_sizes_copy(x, split_sizes, dim)

    assert len(out) == len(ref)
    for actual, expected in zip(out, ref):
        torch.testing.assert_close(actual, expected)
        assert actual.data_ptr() != x.data_ptr() or actual.numel() == 0
