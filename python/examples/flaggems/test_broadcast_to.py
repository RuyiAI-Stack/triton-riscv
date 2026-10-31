import pytest
import torch

from .broadcast_to import broadcast_to


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64, torch.int32]
)
@pytest.mark.parametrize(
    ("shape", "size"),
    [
        ((), (2, 3)),
        ((3,), (2, 3)),
        ((1, 3), [4, 3]),
        ((1, 2, 3), torch.Size([2, 4, 2, 3])),
        ((1,), (1023,)),
        ((1,), (1024,)),
    ],
)
def test_broadcast_to(dtype, shape, size):
    torch.manual_seed(0)
    x = (
        torch.arange(
            int(torch.prod(torch.tensor(shape))), dtype=dtype, device="cpu"
        ).reshape(shape)
        + 1
    )
    tri_out = broadcast_to(x, size)
    ref_out = torch.broadcast_to(x, size)

    torch.testing.assert_close(tri_out, ref_out)
    assert tri_out.data_ptr() != x.data_ptr()
    assert tri_out.is_contiguous()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 2, 3), (2, 3, 4)])
def test_broadcast_to_noop_returns_view(dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    tri_out = broadcast_to(x, shape)
    ref_out = torch.broadcast_to(x, shape)

    torch.testing.assert_close(tri_out, ref_out)
    assert tri_out.data_ptr() == x.data_ptr()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 3)])
def test_broadcast_to_with_negative_size_and_empty(dtype, shape):
    torch.manual_seed(0)
    x = torch.ones(shape, dtype=dtype, device="cpu")
    tri_out = broadcast_to(x, (-1, 3))
    ref_out = torch.broadcast_to(x, (-1, 3))
    torch.testing.assert_close(tri_out, ref_out)

    x_empty = torch.empty((2, 0), dtype=dtype, device="cpu")
    tri_out2 = broadcast_to(x_empty, (-1, 0))
    ref_out2 = torch.broadcast_to(x_empty, (-1, 0))
    torch.testing.assert_close(tri_out2, ref_out2)
    assert tri_out2.shape == ref_out2.shape == (2, 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 3, 4)])
def test_broadcast_to_non_contiguous_input(dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=dtype).transpose(1, 2).unsqueeze(1)
    tri_out = broadcast_to(x, (2, 5, 4, 3))
    ref_out = torch.broadcast_to(x, (2, 5, 4, 3))
    torch.testing.assert_close(tri_out, ref_out)
    assert tri_out.data_ptr() != x.data_ptr()
    assert tri_out.is_contiguous()


def test_broadcast_to_errors():
    x = torch.empty((2, 3), dtype=torch.float32, device="cpu")

    with pytest.raises(RuntimeError, match="fewer dimensions"):
        broadcast_to(x, (3,))

    with pytest.raises(RuntimeError, match="must match the existing size"):
        broadcast_to(x, (4, 4))

    with pytest.raises(TypeError, match="list/tuple/torch.Size"):
        broadcast_to(x, 3)


def test_broadcast_to_more_than_16_dimensions():
    x = torch.ones((1,), dtype=torch.float32, device="cpu")
    size = (1,) * 17

    tri_out = broadcast_to(x, size)
    ref_out = torch.broadcast_to(x, size)

    torch.testing.assert_close(tri_out, ref_out)
