import pytest
import torch

from .roll import roll


@pytest.mark.parametrize("size", [(512,), (16, 256), (4, 64, 128)])
def test_roll_shift(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.roll(x, shifts=1)
    tri_out = roll(x, shifts=1)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [(512,), (16, 256), (4, 64, 128)])
def test_roll_shift_dim0(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.roll(x, shifts=2, dims=0)
    tri_out = roll(x, shifts=2, dims=0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [(16, 256), (4, 64, 128)])
def test_roll_shift_dim1(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.roll(x, shifts=3, dims=1)
    tri_out = roll(x, shifts=3, dims=1)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [(16, 256), (4, 64, 128)])
def test_roll_multi_dim_shifts(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.roll(x, shifts=(2, 1), dims=(0, 1))
    tri_out = roll(x, shifts=(2, 1), dims=(0, 1))

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
