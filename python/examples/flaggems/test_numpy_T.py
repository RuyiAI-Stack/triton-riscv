import pytest
import torch

from .numpy_T import numpy_T


SHAPES = [(), (8,), (512,), (1023,), (1024,), (16, 32), (7, 8, 9), (2, 3, 4, 5)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_numpy_T(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)

    ref = torch.ops.aten.numpy_T(x)
    out = numpy_T(x)

    assert out.dtype == ref.dtype
    assert out.shape == ref.shape
    assert out.stride() == ref.stride()
    assert out.storage_offset() == ref.storage_offset()
    assert torch.equal(out, ref)


@pytest.mark.parametrize("shape", [(5, 6), (2, 3, 4, 5)])
def test_numpy_T_is_view(shape):
    """The result is a zero-copy view: shared storage, writes propagate both ways."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    x_shape, x_stride, x_offset = x.shape, x.stride(), x.storage_offset()

    out = numpy_T(x)

    assert out is not x
    assert out._is_view()
    assert out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()

    out.fill_(1.0)
    assert torch.all(x == 1.0)
    assert x.shape == x_shape
    assert x.stride() == x_stride
    assert x.storage_offset() == x_offset


@pytest.mark.parametrize("shape", [(), (8,)])
def test_numpy_T_low_rank_distinct_object(shape):
    """0-D and 1-D keep their shape but still yield a distinct view object."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)

    out = numpy_T(x)

    assert out is not x
    assert out.shape == x.shape
    assert out._is_view()
    assert out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_numpy_T_non_contiguous(dtype):
    torch.manual_seed(0)
    base = torch.randn((32, 16), dtype=dtype)
    x = base.t()[4:12, 2:14]

    ref = torch.ops.aten.numpy_T(x)
    out = numpy_T(x)

    assert out.shape == ref.shape
    assert out.stride() == ref.stride()
    assert out.storage_offset() == ref.storage_offset()
    assert torch.equal(out, ref)


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (7, 8, 9)])
def test_numpy_T_strided_and_expanded(shape):
    """A zero-stride expand and a stepped slice both read exactly like a clone."""
    torch.manual_seed(0)
    base = torch.randn((1,) + shape[1:], dtype=torch.float32)
    expanded = base.expand(shape)
    assert torch.equal(numpy_T(expanded), torch.ops.aten.numpy_T(expanded))

    stepped = torch.randn((8, 16), dtype=torch.float32)[:, ::2]
    assert torch.equal(numpy_T(stepped), torch.ops.aten.numpy_T(stepped))


def test_numpy_T_lazy_conj_bit_forwarded():
    torch.manual_seed(0)
    base = torch.randn((4, 6), dtype=torch.complex64)
    x = base.conj()
    assert x.is_conj()

    out = numpy_T(x)

    assert out.is_conj()
    assert torch.equal(out, torch.ops.aten.numpy_T(x))


@pytest.mark.parametrize("shape", [(0,), (2, 0), (0, 3, 0)])
def test_numpy_T_empty(shape):
    x = torch.randn(shape, dtype=torch.float32)

    ref = torch.ops.aten.numpy_T(x)
    out = numpy_T(x)

    assert out.shape == ref.shape
    assert out.stride() == ref.stride()
    assert out._is_view()
    assert torch.equal(out, ref)


def test_numpy_T_autograd():
    torch.manual_seed(0)
    x = torch.randn(4, 6, 8, dtype=torch.float32, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)

    out = numpy_T(x)
    grad = torch.randn(out.shape, dtype=torch.float32)
    out.backward(grad)

    ref_out = torch.ops.aten.numpy_T(ref_x)
    ref_out.backward(grad.clone())

    assert torch.equal(x.grad, ref_x.grad)


@pytest.mark.parametrize("bad_input", [3.14, [1.0, 2.0], None])
def test_numpy_T_rejects_non_tensor(bad_input):
    with pytest.raises((RuntimeError, TypeError)):
        numpy_T(bad_input)
