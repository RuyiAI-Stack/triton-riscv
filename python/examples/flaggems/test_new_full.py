import pytest
import torch

from .new_full import new_full


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
def test_new_full_float(shape, fill_value):
    self = torch.empty(0, dtype=torch.float32, device="cpu")
    tri_out = new_full(self, shape, fill_value, dtype=torch.float32, device="cpu")
    ref_out = self.new_full(shape, fill_value, dtype=torch.float32, device="cpu")
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, fill_value",
    [
        ((16, 256), 3),
        ((4, 128), -1),
        ((512,), 42),
    ],
)
def test_new_full_int(shape, fill_value):
    self = torch.empty(0, dtype=torch.int64, device="cpu")
    tri_out = new_full(self, shape, fill_value, dtype=torch.int64, device="cpu")
    ref_out = self.new_full(shape, fill_value, dtype=torch.int64, device="cpu")
    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize(
    "shape, fill_value",
    [
        ((16, 256), True),
        ((4, 128), False),
    ],
)
def test_new_full_bool(shape, fill_value):
    self = torch.empty(0, dtype=torch.bool, device="cpu")
    tri_out = new_full(self, shape, fill_value, dtype=torch.bool, device="cpu")
    ref_out = self.new_full(shape, fill_value, dtype=torch.bool, device="cpu")
    torch.testing.assert_close(tri_out, ref_out)


def test_new_full_invalid_dtype():
    self = torch.empty(0, dtype=torch.int8, device="cpu")
    with pytest.raises(RuntimeError, match="without overflow"):
        new_full(self, (4,), 999999999999, dtype=torch.int8, device="cpu")
