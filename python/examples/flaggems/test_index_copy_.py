import pytest
import torch

from .index_copy_ import index_copy, index_copy_


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((4, 8), 0),
        ((4, 8), 1),
        ((2, 3, 4), 0),
        ((2, 3, 4), 2),
    ],
)
def test_index_copy(shape, dim):
    torch.manual_seed(0)
    inp = torch.randn(*shape, dtype=torch.float32)
    src_shape = list(shape)
    src_shape[dim] = 3
    src = torch.randn(*src_shape, dtype=torch.float32)
    index = torch.randint(0, shape[dim], (src_shape[dim],), dtype=torch.long)

    ref = torch.index_copy(inp, dim, index, src)
    tri = index_copy(inp, dim, index, src)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_index_copy_inplace():
    torch.manual_seed(0)
    inp = torch.randn(4, 8, dtype=torch.float32)
    ref = inp.clone()
    index = torch.tensor([0, 2], dtype=torch.long)
    src = torch.randn(2, 8, dtype=torch.float32)

    ref.index_copy_(0, index, src)
    returned = index_copy_(inp, 0, index, src)

    assert returned is inp
    torch.testing.assert_close(inp, ref, rtol=1e-4, atol=1e-4)
