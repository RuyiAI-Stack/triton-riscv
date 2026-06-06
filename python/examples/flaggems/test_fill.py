import pytest
import torch

from .fill import (
    fill_scalar,
    fill_scalar_,
    fill_scalar_out,
    fill_tensor,
    fill_tensor_,
    fill_tensor_out,
)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
@pytest.mark.parametrize("value", [0.0, 1.5, -3.14])
def test_fill_scalar(shape, dtype, value):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")

    out = fill_scalar(x, value)
    ref_out = torch.full_like(x, value)

    torch.testing.assert_close(out, ref_out)

    # test inplace
    fill_scalar_(x, value)
    torch.testing.assert_close(x, ref_out)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
@pytest.mark.parametrize("value", [0.0, 1.5, -3.14])
def test_fill_tensor(shape, dtype, value):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")
    val_t = torch.tensor(value, dtype=dtype, device="cpu")

    out = fill_tensor(x, val_t)
    ref_out = torch.full_like(x, value)

    torch.testing.assert_close(out, ref_out)

    # test inplace
    fill_tensor_(x, val_t)
    torch.testing.assert_close(x, ref_out)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512,)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("value", [0.0, 1.5])
def test_fill_scalar_out(shape, dtype, value):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")
    out = torch.empty(shape, dtype=dtype, device="cpu")
    ref_out = torch.full_like(x, value)

    result = fill_scalar_out(x, value, out=out)
    torch.testing.assert_close(result, ref_out)
    torch.testing.assert_close(out, ref_out)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512,)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("value", [0.0, 1.5])
def test_fill_tensor_out(shape, dtype, value):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")
    val_t = torch.tensor(value, dtype=dtype, device="cpu")
    out = torch.empty(shape, dtype=dtype, device="cpu")
    ref_out = torch.full_like(x, value)

    result = fill_tensor_out(x, val_t, out=out)
    torch.testing.assert_close(result, ref_out)
    torch.testing.assert_close(out, ref_out)
