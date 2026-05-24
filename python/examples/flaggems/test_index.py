import pytest
import torch

from .index import index


@pytest.mark.parametrize(
    "shape, indices",
    [
        ((64,), (torch.tensor([0, 2, 5, 31, 63]),)),
        ((256,), (torch.tensor([10, 20, 30, 40, 50]),)),
        ((512,), (torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),)),
    ],
)
def test_index_1d(shape, indices):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref_out = x[indices]
    tri_out = index(x, indices)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, indices",
    [
        ((16, 32), (torch.tensor([0, 5, 10, 15]),)),
        ((8, 64), (torch.tensor([2, 4, 6]),)),
    ],
)
def test_index_2d_oneidx(shape, indices):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref_out = x[indices[0]]
    tri_out = index(x, indices)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, indices",
    [
        (
            (16, 32),
            (torch.tensor([0, 5, 10, 15]), torch.tensor([0, 10, 20, 30])),
        ),
        ((8, 16), (torch.tensor([2, 4, 6]), torch.tensor([0, 8, 15]))),
    ],
)
def test_index_2d_twoidx(shape, indices):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref_out = x[indices[0], indices[1]]
    tri_out = index(x, indices)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, indices",
    [
        ((4, 16, 8), (torch.tensor([0, 2, 3]),)),
        ((2, 32, 16), (torch.tensor([0, 1]),)),
    ],
)
def test_index_3d_oneidx(shape, indices):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref_out = x[indices[0]]
    tri_out = index(x, indices)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, indices",
    [
        (
            (4, 16, 8),
            (torch.tensor([0, 2, 3]), torch.tensor([0, 5, 10])),
        ),
    ],
)
def test_index_3d_twoidx(shape, indices):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref_out = x[indices[0], indices[1]]
    tri_out = index(x, indices)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_index_size_variants(size):
    torch.manual_seed(0)
    x = torch.randn((size,), dtype=torch.float32, device="cpu")
    idx = torch.randint(0, size, (min(size // 4, 256),))
    ref_out = x[idx]
    tri_out = index(x, (idx,))
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
