import torch

from .empty import empty


def test_empty_shape_dtype_device():
    out = empty(2, 3, dtype=torch.float32, device="cpu")

    assert out.shape == (2, 3)
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"
