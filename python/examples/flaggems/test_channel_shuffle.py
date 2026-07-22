from math import prod

import pytest
import torch

from .channel_shuffle import channel_shuffle


@pytest.mark.parametrize("shape, groups", [((1, 4, 2, 2), 2), ((2, 6, 3, 5), 3)])
def test_channel_shuffle(shape, groups):
    x = torch.arange(prod(shape), dtype=torch.float32, device="cpu")
    x = x.reshape(shape)

    out = channel_shuffle(x, groups=groups)
    ref = torch.channel_shuffle(x, groups=groups)

    torch.testing.assert_close(out, ref)


def test_channel_shuffle_3d():
    x = torch.arange(2 * 4 * 7, dtype=torch.float32, device="cpu").reshape(2, 4, 7)

    out = channel_shuffle(x, groups=2)
    ref = torch.channel_shuffle(x, groups=2)

    torch.testing.assert_close(out, ref)


def test_channel_shuffle_higher_spatial_rank():
    x = torch.arange(2 * 4 * 3 * 2 * 5, dtype=torch.float32, device="cpu").reshape(
        2, 4, 3, 2, 5
    )

    out = channel_shuffle(x, groups=2)
    ref = torch.channel_shuffle(x, groups=2)

    torch.testing.assert_close(out, ref)
