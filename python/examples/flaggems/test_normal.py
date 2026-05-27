import pytest
import torch

from .normal import (
    normal_,
    normal_float_tensor,
    normal_tensor_float,
    normal_tensor_tensor,
)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
def test_normal_tensor_float(shape):
    torch.manual_seed(0)
    mean = torch.zeros(shape, dtype=torch.float32, device="cpu")
    std = 1.0

    ref_out = torch.normal(mean, std)
    tri_out = normal_tensor_float(mean, std)

    # Check shape and dtype, not exact values (random)
    assert tri_out.shape == ref_out.shape
    assert tri_out.dtype == ref_out.dtype
    # Check mean is approximately 0
    assert abs(tri_out.mean().item()) < 0.5
    # Check std is approximately 1
    assert abs(tri_out.std().item() - 1.0) < 0.5


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,)],
)
def test_normal_float_tensor(shape):
    torch.manual_seed(0)
    mean = 0.0
    std = torch.ones(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.normal(mean, std)
    tri_out = normal_float_tensor(mean, std)

    assert tri_out.shape == ref_out.shape
    assert tri_out.dtype == ref_out.dtype
    assert abs(tri_out.mean().item()) < 0.5
    assert abs(tri_out.std().item() - 1.0) < 0.5


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128)],
)
def test_normal_inplace(shape):
    torch.manual_seed(0)
    self = torch.empty(shape, dtype=torch.float32, device="cpu")

    normal_(self, mean=0, std=1)

    assert self.shape == shape
    assert abs(self.mean().item()) < 0.5
    assert abs(self.std().item() - 1.0) < 0.5


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,)],
)
def test_normal_tensor_tensor(shape):
    torch.manual_seed(0)
    mean = torch.full(shape, 2.0, dtype=torch.float32, device="cpu")
    std = torch.full(shape, 3.0, dtype=torch.float32, device="cpu")

    tri_out = normal_tensor_tensor(mean, std)

    assert tri_out.shape == shape
    assert tri_out.dtype == torch.float32
    assert abs(tri_out.mean().item() - 2.0) < 0.6
    assert abs(tri_out.std().item() - 3.0) < 0.6
