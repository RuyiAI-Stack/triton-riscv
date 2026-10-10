import pytest
import torch

from .remainder import _remainder, remainder, remainder_


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_remainder(shape, dtype):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype, device="cpu")
        y = torch.randn(shape, dtype=dtype, device="cpu")
    else:
        x = torch.randint(-10, 10, shape, dtype=dtype, device="cpu")
        y = torch.randint(1, 10, shape, dtype=dtype, device="cpu")

    tri_out = remainder(x, y)
    ref_out = torch.remainder(x, y)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_remainder_inplace(shape, dtype):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype, device="cpu")
    else:
        x = torch.randint(-10, 10, shape, dtype=dtype, device="cpu")

    y = 3
    ref_out = torch.remainder(x, y)

    tri_out = remainder_(x, y)

    assert tri_out is x
    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test__remainder(shape, dtype):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype, device="cpu")
        y = torch.randn(shape, dtype=dtype, device="cpu")
    else:
        x = torch.randint(-10, 10, shape, dtype=dtype, device="cpu")
        y = torch.randint(1, 10, shape, dtype=dtype, device="cpu")

    tri_out = _remainder(x, y)
    ref_out = torch.remainder(x, y)

    torch.testing.assert_close(tri_out, ref_out)
