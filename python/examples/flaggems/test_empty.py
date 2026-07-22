import pytest
import torch

from .empty import empty


def test_empty_shape_dtype_device():
    out = empty(2, 3, dtype=torch.float32, device="cpu")

    assert out.shape == (2, 3)
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"


@pytest.mark.parametrize("size", [(2, 3), [2, 3], torch.Size([2, 3]), (), (2, 0)])
def test_empty_sequence_shape(size):
    actual = empty(size, dtype=torch.float64, device="cpu")
    expected = torch.empty(size, dtype=torch.float64, device="cpu")
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
