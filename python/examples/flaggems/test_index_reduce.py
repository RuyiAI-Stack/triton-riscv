import pytest
import torch

from .index_reduce import index_reduce_


def _assert_index_reduce_match(
    inp,
    dim,
    index,
    source,
    reduce,
    include_self,
):
    ref_out = inp.clone()
    ref_out.index_reduce_(dim, index, source, reduce, include_self=include_self)

    tri_out = index_reduce_(
        inp.clone(), dim, index, source, reduce, include_self=include_self
    )

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)
    assert tri_out is not ref_out


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("reduce", ["prod", "mean", "amax", "amin"])
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("index_pattern", ["identity", "dup0"])
def test_index_reduce_matches_torch(
    shape, dtype, reduce, include_self, dim, index_pattern
):
    torch.manual_seed(0)
    device = "cpu"
    dim = dim % len(shape)
    inp = torch.randn(shape, device=device, dtype=dtype)
    source = torch.randn(shape, device=device, dtype=dtype)

    index_len = shape[dim]
    if index_len > 0:
        index = torch.arange(index_len, dtype=torch.int64, device=device)
    else:
        index = torch.tensor([], dtype=torch.int64, device=device)

    if index_pattern == "dup0" and index_len > 0:
        index = torch.zeros_like(index)

    _assert_index_reduce_match(
        inp,
        dim,
        index,
        source,
        reduce,
        include_self,
    )
