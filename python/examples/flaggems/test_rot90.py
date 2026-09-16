import pytest
import torch

from .rot90 import rot90


@pytest.mark.parametrize(
    ("shape", "dims"),
    [
        ((3, 4), [0, 1]),
        ((3, 4), [-2, -1]),
        ((2, 3, 4), [0, 1]),
        ((2, 3, 4), [-2, -1]),
        ((2, 3, 4), [1, 2]),
        ((2, 3, 4), [0, 2]),
        ((128, 127), [0, 1]),
        ((128, 127), [-2, -1]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5, -1, -2])
def test_rot90(shape, dims, k, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    tri_out = rot90(x, k=k, dims=dims)
    ref_out = torch.rot90(x, k=k, dims=dims)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(0, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_rot90_empty(shape, dtype):
    torch.manual_seed(0)
    x = torch.empty(shape, device="cpu", dtype=dtype)
    tri_out = rot90(x, k=1, dims=[1, 2])
    ref_out = torch.rot90(x, k=1, dims=[1, 2])
    assert tri_out.shape == ref_out.shape
    assert tri_out.numel() == 0


@pytest.mark.parametrize("shape", [(2, 3, 4), (4, 5, 6)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_rot90_tuple_dims(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")

    tri_out = rot90(x, k=1, dims=(1, 2))
    ref_out = torch.rot90(x, k=1, dims=(1, 2))

    torch.testing.assert_close(tri_out, ref_out)
