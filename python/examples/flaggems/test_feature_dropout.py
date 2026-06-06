import pytest
import torch

from .feature_dropout import feature_dropout, feature_dropout_


@pytest.mark.parametrize(
    "shape",
    [
        (16, 3, 32, 32),
        (4, 16, 8, 8),
        (2, 8, 16),
        (512, 64),
        (1023, 64),
        (1024, 64),
    ],
)
@pytest.mark.parametrize("p", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_feature_dropout(shape, p):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device="cpu")

    # Run triton implementation
    out = feature_dropout(x, p, train=True)

    if p == 1.0:
        assert torch.all(out == 0)
    elif p == 0.0:
        torch.testing.assert_close(out, x)
    else:
        # Check if entire channels are dropped or kept
        # out should be either 0 or x * scale for a whole channel
        scale = 1.0 / (1.0 - p)
        out_flatten = out.view(shape[0], shape[1], -1)
        x_flatten = x.view(shape[0], shape[1], -1)

        for n in range(shape[0]):
            for c in range(shape[1]):
                channel_out = out_flatten[n, c]
                channel_x = x_flatten[n, c]

                # Channel is either all zero or all scaled
                if torch.all(channel_out == 0):
                    pass
                else:
                    torch.testing.assert_close(channel_out, channel_x * scale)


@pytest.mark.parametrize(
    "shape",
    [(4, 16, 8, 8)],
)
@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_feature_dropout_inplace(shape, p):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device="cpu")
    x_copy = x.clone()

    feature_dropout_(x, p, train=True)

    if p == 1.0:
        assert torch.all(x == 0)
    elif p == 0.0:
        torch.testing.assert_close(x, x_copy)
    else:
        scale = 1.0 / (1.0 - p)
        x_flatten = x.view(shape[0], shape[1], -1)
        x_copy_flatten = x_copy.view(shape[0], shape[1], -1)

        for n in range(shape[0]):
            for c in range(shape[1]):
                channel_x = x_flatten[n, c]
                channel_copy = x_copy_flatten[n, c]

                if torch.all(channel_x == 0):
                    pass
                else:
                    torch.testing.assert_close(channel_x, channel_copy * scale)


def test_feature_dropout_repeated_calls_advance_philox_offset():
    torch.manual_seed(0)
    x = torch.ones((128, 32), dtype=torch.float32, device="cpu")

    first = feature_dropout(x, 0.5, train=True)
    second = feature_dropout(x, 0.5, train=True)

    assert not torch.equal(first, second)
