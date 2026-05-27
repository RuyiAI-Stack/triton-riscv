import pytest
import torch

from .full_like import full_like


@pytest.mark.parametrize(
    "shape, fill_value",
    [
        ((16, 256), 3.0),
        ((4, 128), -1.0),
        ((512,), 42.0),
        ((1023,), 0.0),
        ((1024,), 7.0),
    ],
)
def test_full_like_float(shape, fill_value):
    x = torch.randn(shape, device="cpu")
    tri_out = full_like(x, fill_value, dtype=torch.float32, device="cpu")
    ref_out = torch.full_like(x, fill_value, dtype=torch.float32, device="cpu")
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, fill_value",
    [
        ((16, 256), 3),
        ((4, 128), -1),
        ((512,), 42),
    ],
)
def test_full_like_int(shape, fill_value):
    x = torch.randint(0, 100, shape, device="cpu")
    tri_out = full_like(x, fill_value, dtype=torch.int64, device="cpu")
    ref_out = torch.full_like(x, fill_value, dtype=torch.int64, device="cpu")
    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize(
    "shape, fill_value",
    [
        ((16, 256), True),
        ((4, 128), False),
    ],
)
def test_full_like_bool(shape, fill_value):
    x = torch.randint(0, 2, shape, device="cpu", dtype=torch.bool)
    tri_out = full_like(x, fill_value, dtype=torch.bool, device="cpu")
    ref_out = torch.full_like(x, fill_value, dtype=torch.bool, device="cpu")
    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize(
    "shape, fill_value",
    [
        ((16, 256), 3.0),
        ((4, 128), 2.5),
    ],
)
def test_full_like_inherit_dtype(shape, fill_value):
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    tri_out = full_like(x, fill_value)
    ref_out = torch.full_like(x, fill_value)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
