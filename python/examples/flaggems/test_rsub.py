import pytest
import torch

from .rsub import rsub, rsub_scalar, rsub_tensor


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_rsub_tt(shape, alpha):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.sub(y, x, alpha=alpha)
    tri_out = rsub(x, y, alpha=alpha)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_rsub_ts(shape, alpha):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = 5.0

    ref_out = torch.sub(y, x, alpha=alpha)
    tri_out = rsub(x, y, alpha=alpha)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_rsub_st(shape, alpha):
    torch.manual_seed(0)
    x = 5.0
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.sub(y, x, alpha=alpha)
    tri_out = rsub(x, y, alpha=alpha)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_rsub_tensor(shape, alpha):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.sub(y, x, alpha=alpha)
    tri_out = rsub_tensor(x, y, alpha=alpha)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_rsub_scalar(shape, alpha):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    scalar_y = 5.0

    ref_out = torch.sub(scalar_y, x, alpha=alpha)
    tri_out = rsub_scalar(x, scalar_y, alpha=alpha)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
