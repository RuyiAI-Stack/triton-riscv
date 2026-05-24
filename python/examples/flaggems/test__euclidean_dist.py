import pytest
import torch

from ._euclidean_dist import _euclidean_dist


def ref_euclidean_dist(x1, x2):
    x1_norm = (x1**2).sum(dim=1, keepdim=True)
    x2_norm = (x2**2).sum(dim=1, keepdim=True)
    dist = x1_norm + x2_norm.T - 2.0 * torch.mm(x1, x2.T)
    return torch.sqrt(torch.clamp(dist, min=0.0))


@pytest.mark.parametrize("N, M, D", [(8, 10, 16), (16, 8, 32), (32, 64, 128)])
def test_euclidean_dist(N, M, D):
    torch.manual_seed(0)
    x1 = torch.randn(N, D, device="cpu", dtype=torch.float32)
    x2 = torch.randn(M, D, device="cpu", dtype=torch.float32)

    ref = ref_euclidean_dist(x1, x2)
    tri = _euclidean_dist(x1, x2)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_euclidean_dist_shape_assertions():
    x1 = torch.randn(8, 16)
    x2 = torch.randn(10, 32)

    with pytest.raises(AssertionError):
        _euclidean_dist(x1, x2)
