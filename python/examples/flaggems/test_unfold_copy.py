from math import prod

import pytest
import torch

from .unfold_copy import unfold_copy


@pytest.mark.parametrize(
    "shape, dimension, size, step",
    [((4, 5), 1, 3, 1), ((2, 3, 5), 2, 3, 2), ((2, 5, 3), 1, 3, 1)],
)
def test_unfold_copy_matches_torch(shape, dimension, size, step):
    x = torch.arange(prod(shape), dtype=torch.float32, device="cpu")
    x = x.reshape(shape)

    out = unfold_copy(x, dimension=dimension, size=size, step=step)
    ref = torch.ops.aten.unfold_copy(x, dimension, size, step)

    torch.testing.assert_close(out, ref)


def test_unfold_copy_matches_torch_strided_input():
    x = (
        torch.arange(30, dtype=torch.float32, device="cpu")
        .reshape(5, 6)
        .transpose(0, 1)
    )

    out = unfold_copy(x, dimension=1, size=3, step=2)
    ref = torch.ops.aten.unfold_copy(x, 1, 3, 2)

    assert not x.is_contiguous()
    torch.testing.assert_close(out, ref)
