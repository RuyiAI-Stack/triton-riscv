import pytest
import torch

from .index_copy_ import index_copy


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
