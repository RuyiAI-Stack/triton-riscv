import pytest
import torch

from ._safe_softmax import _safe_softmax


def safe_softmax_ref(x, dim=-1, dtype=None):
    # PyTorch's native softmax will output NaN if all elements are -inf.
    # We replicate the safe behavior: output 0.0 if all are -inf.
    res = torch.softmax(x.float(), dim=dim)
    all_neginf = (x == float("-inf")).all(dim=dim, keepdim=True)
    res = torch.where(all_neginf, torch.zeros_like(res), res)
    if dtype is not None:
        res = res.to(dtype)
    else:
        res = res.to(x.dtype)
    return res


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test__safe_softmax_1d(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = safe_softmax_ref(x)
    tri_out = _safe_softmax(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 512), (32, 1023), (4, 1024)])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test__safe_softmax_2d(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = safe_softmax_ref(x, dim=dim)
    tri_out = _safe_softmax(x, dim=dim)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test__safe_softmax_all_neginf():
    torch.manual_seed(0)
    x = torch.full((10, 512), float("-inf"), device="cpu", dtype=torch.float32)

    ref_out = safe_softmax_ref(x)
    tri_out = _safe_softmax(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
