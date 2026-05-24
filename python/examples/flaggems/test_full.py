import pytest
import torch

from .full import full


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
def test_full_float(shape, fill_value):
    tri_out = full(shape, fill_value, dtype=torch.float32, device="cpu")
    ref_out = torch.full(shape, fill_value, dtype=torch.float32, device="cpu")
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, fill_value",
    [
        ((16, 256), 3),
        ((4, 128), -1),
        ((512,), 42),
        ((16, 16), 0),
    ],
)
def test_full_int(shape, fill_value):
    tri_out = full(shape, fill_value, dtype=torch.int64, device="cpu")
    ref_out = torch.full(shape, fill_value, dtype=torch.int64, device="cpu")
    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize(
    "shape, fill_value",
    [
        ((16, 256), True),
        ((4, 128), False),
        ((512,), True),
    ],
)
def test_full_bool(shape, fill_value):
    tri_out = full(shape, fill_value, dtype=torch.bool, device="cpu")
    ref_out = torch.full(shape, fill_value, dtype=torch.bool, device="cpu")
    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize(
    "shape, fill_value",
    [
        ((16, 256), 3.0),
        ((4, 128), -1.0),
        ((512,), 3.14),
    ],
)
def test_full_autodtype_float(shape, fill_value):
    tri_out = full(shape, fill_value, device="cpu")
    ref_out = torch.full(shape, fill_value, device="cpu")
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_full_invalid_dtype():
    with pytest.raises(RuntimeError, match="without overflow"):
        full((4,), 999999999999, dtype=torch.int8, device="cpu")
