from math import prod

import pytest
import torch

from .view_copy import view_copy


@pytest.mark.parametrize(
    "shape, size", [((12,), (3, 4)), ((3, 4), (-1, 3)), ((4, 3), (2, 6))]
)
def test_view_copy_matches_torch(shape, size):
    x = torch.arange(prod(shape), dtype=torch.float32, device="cpu")
    x = x.reshape(shape)
    if shape == (4, 3):
        x = x.transpose(0, 1)

    out = view_copy(x, size)
    ref = torch.ops.aten.view_copy(x, size)

    torch.testing.assert_close(out, ref)
    assert out.data_ptr() != x.data_ptr()
