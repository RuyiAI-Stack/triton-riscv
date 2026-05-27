import pytest
import torch

from .topk import topk


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_topk(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_vals, ref_idxs = torch.topk(x, k=10)
    tri_vals, tri_idxs = topk(x, k=10)

    torch.testing.assert_close(tri_vals, ref_vals, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_idxs, ref_idxs)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_topk_largest_false(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_vals, ref_idxs = torch.topk(x, k=10, largest=False)
    tri_vals, tri_idxs = topk(x, k=10, largest=False)

    torch.testing.assert_close(tri_vals, ref_vals, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_idxs, ref_idxs)
