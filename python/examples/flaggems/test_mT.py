import pytest
import torch

from .mT import mT


SHAPES = [
    (16, 32),
    (512, 1024),
    (7, 8, 1023),
    (2, 3, 4, 5),
    (2, 3, 512, 1024),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mT(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)

    ref = torch.ops.aten.mT(x)
    out = mT(x)

    assert out.dtype == ref.dtype
    assert out.shape == ref.shape
    assert out.stride() == ref.stride()
    assert out.storage_offset() == ref.storage_offset()
    assert torch.equal(out, ref)


@pytest.mark.parametrize("shape", [(5, 6), (2, 3, 4, 5)])
def test_mT_is_view(shape):
    """The result is a zero-copy view: shared storage, writes propagate both ways."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    x_shape, x_stride, x_offset = x.shape, x.stride(), x.storage_offset()

    out = mT(x)

    assert out is not x
    assert out._is_view()
    assert out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()

    out.fill_(1.0)
    assert torch.all(x == 1.0)
    assert x.shape == x_shape
    assert x.stride() == x_stride
    assert x.storage_offset() == x_offset


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mT_non_contiguous(dtype):
    torch.manual_seed(0)
    base = torch.randn((32, 16), dtype=dtype)
    x = base.t()[4:12, 2:14]

    ref = torch.ops.aten.mT(x)
    out = mT(x)

    assert out.shape == ref.shape
    assert out.stride() == ref.stride()
    assert out.storage_offset() == ref.storage_offset()
    assert torch.equal(out, ref)


def test_mT_lazy_conj_bit_forwarded():
    torch.manual_seed(0)
    base = torch.randn((4, 6), dtype=torch.complex64)
    x = base.conj()
    assert x.is_conj()

    out = mT(x)

    assert out.is_conj()
    assert torch.equal(out, torch.ops.aten.mT(x))


@pytest.mark.parametrize("shape", [(0, 5), (2, 0, 3)])
def test_mT_empty(shape):
    x = torch.randn(shape, dtype=torch.float32)

    ref = torch.ops.aten.mT(x)
    out = mT(x)

    assert out.shape == ref.shape
    assert out.stride() == ref.stride()
    assert out._is_view()
    assert torch.equal(out, ref)


def test_mT_autograd():
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=torch.float32, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)

    out = mT(x)
    grad = torch.randn(out.shape, dtype=torch.float32)
    out.backward(grad)

    ref_out = torch.ops.aten.mT(ref_x)
    ref_out.backward(grad.clone())

    assert torch.equal(x.grad, ref_x.grad)


def test_mT_rank0_is_identity():
    x = torch.tensor(3.14, dtype=torch.float32)

    out = mT(x)

    assert out is x


def test_mT_rank1_raises():
    x = torch.randn(8, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        mT(x)


@pytest.mark.parametrize("bad_input", [3.14, [1.0, 2.0], None])
def test_mT_rejects_non_tensor(bad_input):
    with pytest.raises((RuntimeError, TypeError)):
        mT(bad_input)
