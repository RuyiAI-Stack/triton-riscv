import pytest
import torch

from .vstack import vstack


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_vstack(size):
    torch.manual_seed(0)
    a = torch.randn(2, size, device="cpu", dtype=torch.float32)
    b = torch.randn(3, size, device="cpu", dtype=torch.float32)

    ref_out = torch.vstack([a, b])
    tri_out = vstack([a, b])

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_vstack_single(size):
    torch.manual_seed(0)
    a = torch.randn(4, size, device="cpu", dtype=torch.float32)

    ref_out = torch.vstack([a])
    tri_out = vstack([a])

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_vstack_many(size):
    torch.manual_seed(0)
    tensors = [
        torch.randn(1, size, device="cpu", dtype=torch.float32)
        for _ in range(6)
    ]

    ref_out = torch.vstack(tensors)
    tri_out = vstack(tensors)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
