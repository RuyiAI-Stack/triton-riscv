import pytest
import torch

from .__ilshift__ import __ilshift__


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test___ilshift__(dtype, shape):
    torch.manual_seed(0)
    x = torch.arange(shape[0], dtype=dtype, device="cpu")
    x_ref = x.clone()
    other = torch.tensor(2, dtype=dtype, device="cpu")

    tri_out = __ilshift__(x, other)
    ref_out = x_ref << other

    assert tri_out is x
    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(3, 4), (4, 4)])
def test___ilshift__broadcast_other(dtype, shape):
    torch.manual_seed(0)
    x = torch.arange(shape[0] * shape[1], dtype=dtype, device="cpu").reshape(shape)
    x_ref = x.clone()
    other = torch.arange(shape[1], dtype=dtype, device="cpu")

    tri_out = __ilshift__(x, other)
    ref_out = x_ref << other
    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(4, 3), (5, 6)])
def test___ilshift___non_contiguous_input(dtype, shape):
    torch.manual_seed(0)
    x = (
        torch.arange(shape[0] * shape[1], dtype=dtype, device="cpu")
        .reshape(shape)
        .transpose(0, 1)
    )
    x_ref = x.clone()
    other = 3

    tri_out = __ilshift__(x, other)
    ref_out = x_ref << other

    assert tri_out is x
    torch.testing.assert_close(tri_out, ref_out)
