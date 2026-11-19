import pytest
import torch

from .__irshift__ import __irshift__

INT_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]


@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test___irshift__(shape, dtype):
    torch.manual_seed(0)
    bits = torch.iinfo(dtype).bits
    x = torch.randint(-(2 ** (bits - 2)), 2 ** (bits - 2), shape, dtype=dtype)
    shift = torch.randint(0, bits, shape, dtype=dtype)

    ref = x.clone()
    ref.__irshift__(shift)

    tri = x.clone()
    ret = __irshift__(tri, shift)

    assert ret is tri, "__irshift__ must return the same tensor"
    assert torch.equal(tri, ref)


@pytest.mark.parametrize("shape", [(0,), (0, 8)])
def test___irshift___empty(shape):
    torch.manual_seed(0)
    tri = torch.empty(shape, dtype=torch.int32)
    shift = torch.empty(shape, dtype=torch.int32)

    ref = tri.clone()
    ref.__irshift__(shift)

    ret = __irshift__(tri, shift)

    assert ret is tri
    assert tri.shape == ref.shape
    assert torch.equal(tri, ref)


def test___irshift___non_contiguous():
    torch.manual_seed(0)
    x = torch.randint(-32, 32, (32, 64), dtype=torch.int32)
    shift = torch.randint(0, 32, (32, 64), dtype=torch.int32)

    ref_base = x.clone()
    ref_view = ref_base.t()
    ref_view.__irshift__(shift.t())

    tri_base = x.clone()
    tri_view = tri_base.t()
    assert not tri_view.is_contiguous()

    ret = __irshift__(tri_view, shift.t())

    assert ret is tri_view
    assert torch.equal(tri_base, ref_base), "in-place must write through the view"


def test___irshift___scalar():
    torch.manual_seed(0)
    x = torch.randint(-32, 32, (512,), dtype=torch.int32)

    ref = x.clone()
    ref.__irshift__(3)

    tri = x.clone()
    ret = __irshift__(tri, 3)

    assert ret is tri
    assert torch.equal(tri, ref)


def test___irshift___other_wider_dtype():
    torch.manual_seed(0)
    x = torch.randint(-32, 32, (512,), dtype=torch.int32)
    shift = torch.randint(0, 32, (512,), dtype=torch.int64)

    ref = x.clone()
    ref.__irshift__(shift)

    tri = x.clone()
    ret = __irshift__(tri, shift)

    assert ret is tri
    assert tri.dtype == torch.int32
    assert torch.equal(tri, ref)


def test___irshift___broadcast_other():
    torch.manual_seed(0)
    x = torch.randint(-32, 32, (16, 64), dtype=torch.int32)
    shift = torch.randint(0, 32, (64,), dtype=torch.int32)

    ref = x.clone()
    ref.__irshift__(shift)

    tri = x.clone()
    ret = __irshift__(tri, shift)

    assert ret is tri
    assert torch.equal(tri, ref)


def test___irshift___rejects_non_integer():
    torch.manual_seed(0)
    x = torch.randn(64)
    shift = torch.randint(0, 32, (64,), dtype=torch.int32)

    with pytest.raises(TypeError):
        __irshift__(x, shift)

    xb = torch.tensor([True, False] * 32)
    with pytest.raises(TypeError):
        __irshift__(xb, xb)
