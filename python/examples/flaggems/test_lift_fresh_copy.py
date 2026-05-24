import pytest
import torch

from .lift_fresh_copy import lift_fresh_copy, lift_fresh_copy_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_lift_fresh_copy(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = x.clone()
    tri_out = lift_fresh_copy(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
def test_lift_fresh_copy_2d(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = x.clone()
    tri_out = lift_fresh_copy(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    assert tri_out.data_ptr() != x.data_ptr()


@pytest.mark.parametrize("shape", [(512,), (1024,)])
def test_lift_fresh_copy_out(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    out = torch.empty(shape, dtype=torch.float32, device="cpu")

    ref_out = x.clone()
    tri_out = lift_fresh_copy_out(x, out=out)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1024,)])
def test_lift_fresh_copy_out_no_out(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = x.clone()
    tri_out = lift_fresh_copy_out(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
