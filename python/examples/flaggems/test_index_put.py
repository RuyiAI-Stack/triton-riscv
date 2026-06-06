import pytest
import torch

from .index_put import index_put, index_put_


@pytest.mark.parametrize(
    "shape, idx, val_shape",
    [
        ((8,), [3], (1,)),
        ((16,), [0, 5, 10], (3,)),
        ((4, 8), [0, 2], (2, 8)),
    ],
)
def test_index_put_1d_index(shape, idx, val_shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    idx_t = torch.tensor(idx, dtype=torch.long, device="cpu")
    vals = torch.randn(val_shape, dtype=torch.float32, device="cpu")

    x_ref[idx_t] = vals
    index_put_(x, (idx_t,), vals)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, idx",
    [
        ((4, 8), ([0, 2], [1, 5])),
        ((4, 8), ([3, 1], [2, 7])),
    ],
)
def test_index_put_2d_indices(shape, idx):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    idx_t = [torch.tensor(i, dtype=torch.long, device="cpu") for i in idx]
    vals = torch.randn(2, dtype=torch.float32, device="cpu")

    x_ref[idx_t[0], idx_t[1]] = vals
    index_put_(x, tuple(idx_t), vals)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, idx",
    [
        ((8,), [3]),
        ((16,), [0, 5, 10]),
        ((4, 8), [0, 2]),
    ],
)
def test_index_put(shape, idx):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    if x.ndim == 1:
        idx_t = torch.tensor(idx, dtype=torch.long, device="cpu")
        vals = torch.randn(len(idx), dtype=torch.float32, device="cpu")
        x_ref[idx_t] = vals
        out = index_put(x, (idx_t,), vals)
    else:
        idx_t = torch.tensor(idx, dtype=torch.long, device="cpu")
        vals = torch.randn(len(idx), shape[1], dtype=torch.float32, device="cpu")
        x_ref[idx_t] = vals
        out = index_put(x, (idx_t,), vals)
    torch.testing.assert_close(out, x_ref, rtol=1e-4, atol=1e-4)


def test_index_put_accumulate():
    torch.manual_seed(0)
    x = torch.randn(8, dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    idx_t = torch.tensor([0, 2, 5], dtype=torch.long, device="cpu")
    vals = torch.randn(3, dtype=torch.float32, device="cpu")

    x_ref.index_add_(0, idx_t, vals)
    index_put_(x, (idx_t,), vals, accumulate=True)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, idx",
    [
        ((4, 6), ([0, 2],)),
        ((3, 5), ([1, 1],)),
    ],
)
def test_index_put_inplace(shape, idx):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    idx_t = torch.tensor(idx[0], dtype=torch.long, device="cpu")
    vals = torch.randn(len(idx[0]), shape[1], dtype=torch.float32, device="cpu")

    x_ref[idx_t] = vals
    index_put_(x, (idx_t,), vals)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
