import pytest
import torch

from .masked_fill import masked_fill, masked_fill_


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
def test_masked_fill(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device="cpu")
    value = 0.0

    ref_out = torch.masked_fill(x, mask, value)
    tri_out = masked_fill(x, mask, value)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
def test_masked_fill_value(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device="cpu")
    value = 3.14

    ref_out = torch.masked_fill(x, mask, value)
    tri_out = masked_fill(x, mask, value)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128)],
)
def test_masked_fill_inplace(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device="cpu")
    value = 0.0
    x_ref = x.clone()

    x_ref.masked_fill_(mask, value)
    masked_fill_(x, mask, value)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
