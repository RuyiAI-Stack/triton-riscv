import pytest
import torch

from .floor_ import floor_


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
def test_floor_(shape):
    torch.manual_seed(0)
    device = "cpu"

    x = torch.randn(shape, dtype=torch.float32, device=device) * 10.0
    x_ref = x.clone()

    x_ref.floor_()
    floor_(x)

    torch.testing.assert_close(x, x_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(16, 256), (1023,)])
def test_floor_non_contiguous(shape):
    torch.manual_seed(0)
    device = "cpu"

    x = torch.randn(shape, dtype=torch.float32, device=device) * 10.0
    if len(shape) > 1:
        x = x[:, ::2]
    else:
        x = x[::2]

    x_ref = x.clone()

    x_ref.floor_()
    floor_(x)

    torch.testing.assert_close(x, x_ref, rtol=1e-5, atol=1e-5)
