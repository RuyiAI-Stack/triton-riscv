import pytest
import torch

from .where import (
    where,
    where_scalar_other,
    where_scalar_self,
    where_self,
    where_self_out,
)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_where(shape):
    torch.manual_seed(0)
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device="cpu")
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.where(cond, x, y)
    tri_out = where(cond, x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_where_scalar(shape):
    torch.manual_seed(0)
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device="cpu")
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.where(cond, x, 0.0)
    tri_out = where(cond, x, 0.0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_where_self(shape):
    torch.manual_seed(0)
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device="cpu")
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.where(cond, x, y)
    tri_out = where_self(cond, x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_where_scalar_self(shape):
    torch.manual_seed(0)
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device="cpu")
    scalar_self = 1.5
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.where(cond, scalar_self, y)
    tri_out = where_scalar_self(cond, scalar_self, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_where_scalar_other(shape):
    torch.manual_seed(0)
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device="cpu")
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    scalar_other = 0.0

    ref_out = torch.where(cond, x, scalar_other)
    tri_out = where_scalar_other(cond, x, scalar_other)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_where_self_out(shape):
    torch.manual_seed(0)
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device="cpu")
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")
    out = torch.empty(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.where(cond, x, y)
    tri_out = where_self_out(cond, x, y, out=out)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    # Also verify the out tensor was updated in-place
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
