import pytest
import torch

from .add import add, add_


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
@pytest.mark.parametrize("alpha", [1.0, 2.5])
def test_add_tt(shape, alpha):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.add(x, y, alpha=alpha)
    tri_out = add(x, y, alpha=alpha)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
@pytest.mark.parametrize("alpha", [1.0, 2.5])
def test_add_ts(shape, alpha):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = 5.0

    ref_out = torch.add(x, y, alpha=alpha)
    tri_out = add(x, y, alpha=alpha)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
@pytest.mark.parametrize("alpha", [1.0, 2.5])
def test_add_st(shape, alpha):
    torch.manual_seed(0)
    x = 5.0
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.add(x, y, alpha=alpha)
    tri_out = add(x, y, alpha=alpha)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape_A, shape_B", [((16, 256), (256,)), ((4, 1), (4, 128))])
@pytest.mark.parametrize("alpha", [1.0, 2.5])
def test_add_broadcast(shape_A, shape_B, alpha):
    torch.manual_seed(0)
    x = torch.randn(shape_A, dtype=torch.float32, device="cpu")
    y = torch.randn(shape_B, dtype=torch.float32, device="cpu")

    ref_out = torch.add(x, y, alpha=alpha)
    tri_out = add(x, y, alpha=alpha)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
def test_add_complex(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.complex64, device="cpu")
    y = torch.randn(shape, dtype=torch.complex64, device="cpu")

    ref_out = torch.add(x, y, alpha=1.0)
    tri_out = add(x, y, alpha=1.0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256)])
def test_add_inplace(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")
    x_ref = x.clone()

    x_ref.add_(y, alpha=2.0)
    add_(x, y, alpha=2.0)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
