import pytest
import torch

from .sub import sub, sub_


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023,), (1024,)])
@pytest.mark.parametrize("alpha", [1.0, 2.5])
def test_sub_tt(shape, alpha):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.sub(x, y, alpha=alpha)
    tri_out = sub(x, y, alpha=alpha)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023,), (1024,)])
@pytest.mark.parametrize("alpha", [1.0, 2.5])
def test_sub_ts(shape, alpha):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = 5.0

    ref_out = torch.sub(x, y, alpha=alpha)
    tri_out = sub(x, y, alpha=alpha)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023,), (1024,)])
@pytest.mark.parametrize("alpha", [1.0, 2.5])
def test_sub_st(shape, alpha):
    torch.manual_seed(0)
    x = 5.0
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.sub(x, y, alpha=alpha)
    tri_out = sub(x, y, alpha=alpha)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape_A, shape_B", [((16, 256), (256,)), ((4, 1), (4, 128))])
@pytest.mark.parametrize("alpha", [1.0, 2.5])
def test_sub_broadcast(shape_A, shape_B, alpha):
    torch.manual_seed(0)
    x = torch.randn(shape_A, dtype=torch.float32, device="cpu")
    y = torch.randn(shape_B, dtype=torch.float32, device="cpu")

    ref_out = torch.sub(x, y, alpha=alpha)
    tri_out = sub(x, y, alpha=alpha)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
def test_sub_complex(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.complex64, device="cpu")
    y = torch.randn(shape, dtype=torch.complex64, device="cpu")

    ref_out = torch.sub(x, y, alpha=1.0)
    tri_out = sub(x, y, alpha=1.0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (1023,)])
def test_sub_inplace(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")
    x_ref = x.clone()

    x_ref.sub_(y, alpha=2.0)
    sub_(x, y, alpha=2.0)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
