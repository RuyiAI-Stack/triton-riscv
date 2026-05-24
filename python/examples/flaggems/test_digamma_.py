import pytest
import torch

from .digamma_ import digamma_


@pytest.mark.parametrize(
    "shape",
    [
        (16, 256),
        (4, 128),
        # Test required sizes: 512, 1023, 1024
        (512,),
        (1023,),
        (1024,),
    ],
)
def test_digamma_(shape):
    torch.manual_seed(0)
    device = "cpu"

    x = torch.rand(shape, dtype=torch.float32, device=device) * 10.0 + 0.1
    x_ref = x.clone().detach()

    ref_out = torch.digamma(x_ref)

    # In-place operation
    digamma_(x)

    torch.testing.assert_close(x, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [
        (16, 256),
        (1023,),
    ],
)
def test_digamma_non_contiguous(shape):
    torch.manual_seed(0)
    device = "cpu"

    # Make non-contiguous by slicing
    x = torch.rand(shape, dtype=torch.float32, device=device) * 10.0 + 0.1
    if len(shape) > 1:
        x = x[:, ::2]
    else:
        x = x[::2]

    x_ref = x.clone().detach()

    ref_out = torch.digamma(x_ref)

    # In-place operation
    digamma_(x)

    torch.testing.assert_close(x, ref_out, rtol=1e-3, atol=1e-3)
