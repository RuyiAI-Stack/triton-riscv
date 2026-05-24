import pytest
import torch

from .index_select import index_select


@pytest.mark.parametrize("shape", [(64, 128), (32, 16)])
@pytest.mark.parametrize("dim", [0, 1])
def test_index_select_2d(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    index = torch.randint(0, shape[dim], (16,), device="cpu")

    ref_out = torch.index_select(x, dim, index)
    tri_out = index_select(x, dim, index)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(8, 16, 32)])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_index_select_3d(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    index = torch.randint(0, shape[dim], (10,), device="cpu")

    ref_out = torch.index_select(x, dim, index)
    tri_out = index_select(x, dim, index)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(32, 64)])
def test_index_select_all_elements(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    index = torch.arange(shape[1], device="cpu")

    ref_out = torch.index_select(x, 1, index)
    tri_out = index_select(x, 1, index)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_index_select_single_element():
    x = torch.randn(32, 64, dtype=torch.float32, device="cpu")
    index = torch.tensor([5], device="cpu")

    ref_out = torch.index_select(x, 0, index)
    tri_out = index_select(x, 0, index)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_index_select_negative_dim():
    torch.manual_seed(0)
    x = torch.randn(16, 32, dtype=torch.float32, device="cpu")
    index = torch.randint(0, 16, (8,), device="cpu")

    ref_out = torch.index_select(x, -2, index)
    tri_out = index_select(x, -2, index)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(128, 64)])
def test_index_select_large(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    index = torch.randint(0, shape[0], (512,), device="cpu")

    ref_out = torch.index_select(x, 0, index)
    tri_out = index_select(x, 0, index)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
