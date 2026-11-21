import pytest
import torch

from .__rshift__ import __rshift__, __rshift___out

INT_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]


@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test___rshift__(shape, dtype):
    torch.manual_seed(0)
    bits = torch.iinfo(dtype).bits
    x = torch.randint(-(2 ** (bits - 2)), 2 ** (bits - 2), shape, dtype=dtype)
    shift = torch.randint(0, bits, shape, dtype=dtype)

    ref = x >> shift
    tri = __rshift__(x, shift)

    assert tri.dtype == ref.dtype
    assert torch.equal(tri, ref)


@pytest.mark.parametrize("shape", [(0,), (0, 8)])
def test___rshift___empty(shape):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=torch.int32)
    shift = torch.empty(shape, dtype=torch.int32)

    ref = x >> shift
    tri = __rshift__(x, shift)

    assert tri.shape == ref.shape
    assert torch.equal(tri, ref)


def test___rshift___non_contiguous():
    torch.manual_seed(0)
    x = torch.randint(-32, 32, (32, 64), dtype=torch.int32)
    shift = torch.randint(0, 32, (32, 64), dtype=torch.int32)
    x_view = x.t()
    shift_view = shift.t()
    assert not x_view.is_contiguous()

    ref = x_view >> shift_view
    tri = __rshift__(x_view, shift_view)

    assert torch.equal(tri, ref)


@pytest.mark.parametrize(
    "shape_x, shape_shift",
    [((16, 64), (1, 64)), ((64,), (16, 64))],
)
def test___rshift___broadcast(shape_x, shape_shift):
    torch.manual_seed(0)
    x = torch.randint(-8, 8, shape_x, dtype=torch.int32)
    shift = torch.randint(0, 32, shape_shift, dtype=torch.int32)

    ref = x >> shift
    tri = __rshift__(x, shift)

    assert torch.equal(tri, ref)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test___rshift___scalar(dtype):
    torch.manual_seed(0)
    bits = torch.iinfo(dtype).bits
    x = torch.randint(-(2 ** (bits - 2)), 2 ** (bits - 2), (512,), dtype=dtype)

    ref = x >> 2
    tri = __rshift__(x, 2)

    assert tri.dtype == ref.dtype
    assert torch.equal(tri, ref)


def test___rshift___out():
    torch.manual_seed(0)
    x = torch.randint(-32, 32, (512,), dtype=torch.int32)
    shift = torch.randint(0, 32, (512,), dtype=torch.int32)
    ref = x >> shift

    out = torch.empty(512, dtype=torch.int32)
    ret = __rshift__(x, shift, out=out)

    assert ret is out, "__rshift__ must return the caller's out tensor"
    assert torch.equal(out, ref)


def test___rshift___out_variant():
    torch.manual_seed(0)
    x = torch.randint(-32, 32, (512,), dtype=torch.int32)
    shift = torch.randint(0, 32, (512,), dtype=torch.int32)
    ref = x >> shift

    out = torch.empty(512, dtype=torch.int32)
    ret = __rshift___out(x, shift, out=out)

    assert ret is out, "__rshift___out must return the caller's out tensor"
    assert torch.equal(out, ref)


def test___rshift___rejects_non_integer():
    torch.manual_seed(0)
    x = torch.randn(64)
    shift = torch.randint(0, 32, (64,), dtype=torch.int32)

    with pytest.raises(TypeError):
        __rshift__(x, shift)

    xb = torch.tensor([True, False] * 32)
    with pytest.raises(TypeError):
        __rshift__(xb, xb)
