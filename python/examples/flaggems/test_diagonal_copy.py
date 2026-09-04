import pytest
import torch

from .diagonal_copy import diagonal_copy


@pytest.mark.parametrize(
    ("shape", "offset", "dim1", "dim2"),
    [
        ((5, 5), 0, 0, 1),
        ((5, 5), 1, 0, 1),
        ((5, 5), -1, 0, 1),
        ((2, 3, 4), 0, 1, 2),
        ((2, 3, 4), 2, 1, 2),
        ((2, 3, 4), -1, 0, 2),
        ((6, 6, 6), 2, 0, 2),
        ((64, 65), 5, 0, 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int32])
def test_diagonal_copy(shape, offset, dim1, dim2, dtype):
    torch.manual_seed(0)
    x = (
        torch.randn(shape, device="cpu").to(dtype)
        if dtype.is_floating_point
        else torch.randint(-10, 10, shape, device="cpu", dtype=dtype)
    )
    x_ref = x.clone()
    tri_out = diagonal_copy(x, offset=offset, dim1=dim1, dim2=dim2)
    ref_out = torch.diagonal(x, offset=offset, dim1=dim1, dim2=dim2).clone()

    torch.testing.assert_close(tri_out, ref_out)
    assert tri_out._base is None

    # mutation on output should not write back to input
    out_clone = tri_out.clone()
    out_clone.add_(1)
    torch.testing.assert_close(x, x_ref)


@pytest.mark.parametrize("shape", [(2, 3, 4), (4, 5, 6)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_diagonal_copy_non_contiguous_input(shape, dtype):
    torch.manual_seed(0)
    x = (
        torch.randn(shape[0] * shape[1] * shape[2], dtype=dtype, device="cpu")
        .reshape(shape)
        .transpose(0, 2)
    )
    tri_out = diagonal_copy(x, offset=-1, dim1=0, dim2=1)
    ref_out = torch.diagonal(x, offset=-1, dim1=0, dim2=1).clone()
    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(3, 4), (5, 6)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_diagonal_copy_empty_and_invalid_dims(shape, dtype):
    torch.manual_seed(0)
    x = torch.arange(shape[0] * shape[1], dtype=dtype, device="cpu").reshape(shape)
    tri_out = diagonal_copy(x, offset=10, dim1=0, dim2=1)
    ref_out = torch.diagonal(x, offset=10, dim1=0, dim2=1).clone()
    torch.testing.assert_close(tri_out, ref_out)
    assert tri_out.numel() == 0
