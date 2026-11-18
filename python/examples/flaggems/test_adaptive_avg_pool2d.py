import pytest
import torch
import torch.nn.functional as F

from .adaptive_avg_pool2d import adaptive_avg_pool2d


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 2, 8, 8), (1, 1, 31, 33)])
@pytest.mark.parametrize("output_size", [(4, 5), [4, 5], 4])
def test_adaptive_avg_pool2d(dtype, shape, output_size):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    out_size = (
        tuple(output_size)
        if isinstance(output_size, (list, tuple))
        else (output_size, output_size)
    )

    tri_out = adaptive_avg_pool2d(x, output_size)
    ref_out = F.adaptive_avg_pool2d(x, out_size)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("output_size", [(6, 1), (1, 7), (2, 2)])
def test_adaptive_avg_pool2d_non_contiguous(dtype, output_size):
    torch.manual_seed(0)
    x = torch.randn(2, 2, 9, 10, device="cpu", dtype=dtype).transpose(2, 3)
    ref_out = F.adaptive_avg_pool2d(x, output_size)
    tri_out = adaptive_avg_pool2d(x, output_size)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 2, 9, 10)])
def test_adaptive_avg_pool2d_scalar_output_size(dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=dtype)
    tri_out = adaptive_avg_pool2d(x, 4)
    ref_out = F.adaptive_avg_pool2d(x, (4, 4))
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 2, 9, 10)])
def test_adaptive_avg_pool2d_zero_output_size(dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=dtype)
    tri_out = adaptive_avg_pool2d(x, (0, 4))
    ref_out = F.adaptive_avg_pool2d(x, (0, 4))
    assert tri_out.shape == ref_out.shape == (2, 2, 0, 4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adaptive_avg_pool2d_unbatched_input(dtype):
    torch.manual_seed(0)
    x = torch.randn(3, 5, 7, device="cpu", dtype=dtype)

    tri_out = adaptive_avg_pool2d(x, (2, 3))
    ref_out = F.adaptive_avg_pool2d(x, (2, 3))

    assert tri_out.shape == (3, 2, 3)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
