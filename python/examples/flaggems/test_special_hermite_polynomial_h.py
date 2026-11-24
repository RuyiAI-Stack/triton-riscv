import pytest
import torch

from .special_hermite_polynomial_h import (
    special_hermite_polynomial_h,
    special_hermite_polynomial_h_tensor_tensor,
)


def _hermite_tolerances(dtype, n):
    if dtype is torch.float32 and (not isinstance(n, int) or n >= 6):
        return 2e-2, 1e-1
    return 1e-4, 1e-4


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", range(10))
def test_special_hermite_polynomial_h(shape, dtype, n):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")

    tri_out = special_hermite_polynomial_h(x, n)
    ref_out = torch.special.hermite_polynomial_h(x, n)

    rtol, atol = _hermite_tolerances(dtype, n)
    torch.testing.assert_close(tri_out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_hermite_polynomial_h_tensor_tensor(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    n = torch.arange(x.numel(), dtype=torch.int64, device="cpu").remainder(10)
    n = n.reshape_as(x)

    tri_out = special_hermite_polynomial_h_tensor_tensor(x, n)
    ref_out = torch.special.hermite_polynomial_h(x, n)

    rtol, atol = _hermite_tolerances(dtype, n)
    torch.testing.assert_close(tri_out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_hermite_polynomial_h_broadcast_degree(dtype):
    torch.manual_seed(0)
    x = torch.randn((17, 33), dtype=dtype, device="cpu")
    n = torch.arange(17, dtype=torch.int64, device="cpu").remainder(10).view(17, 1)

    tri_out = special_hermite_polynomial_h_tensor_tensor(x, n)
    ref_out = torch.special.hermite_polynomial_h(x, n)

    rtol, atol = _hermite_tolerances(dtype, n)
    torch.testing.assert_close(tri_out, ref_out, rtol=rtol, atol=atol)


def test_special_hermite_polynomial_h_non_contiguous_input():
    torch.manual_seed(0)
    x = torch.randn((17, 32), dtype=torch.float32, device="cpu").transpose(0, 1)

    tri_out = special_hermite_polynomial_h(x, 9)
    ref_out = torch.special.hermite_polynomial_h(x, 9)

    rtol, atol = _hermite_tolerances(x.dtype, 9)
    torch.testing.assert_close(tri_out, ref_out, rtol=rtol, atol=atol)


def test_special_hermite_polynomial_h_empty_tensor():
    x = torch.empty((0, 17), dtype=torch.float32, device="cpu")

    tri_out = special_hermite_polynomial_h(x, 3)
    ref_out = torch.special.hermite_polynomial_h(x, 3)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("n", [-1, 10])
def test_special_hermite_polynomial_h_rejects_unsupported_scalar_degree(n):
    x = torch.randn((16, 33), dtype=torch.float32, device="cpu")

    with pytest.raises(ValueError):
        special_hermite_polynomial_h(x, n)


def test_special_hermite_polynomial_h_rejects_unsupported_tensor_degree():
    x = torch.randn((8,), dtype=torch.float32, device="cpu")
    n = torch.tensor([0, 1, 2, 3, 4, 5, 6, 10], dtype=torch.int64, device="cpu")

    with pytest.raises(ValueError):
        special_hermite_polynomial_h_tensor_tensor(x, n)
