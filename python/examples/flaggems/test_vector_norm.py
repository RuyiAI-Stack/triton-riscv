import pytest
import torch

from .vector_norm import vector_norm


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_vector_norm_l2(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32, device="cpu")

    ref = torch.linalg.vector_norm(x, ord=2)
    tri = vector_norm(x, ord=2)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


def test_vector_norm_l2_multi_dim():
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float32, device="cpu")

    ref = torch.linalg.vector_norm(x, ord=2, dim=1)
    tri = vector_norm(x, ord=2, dim=[1])

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


def test_vector_norm_l2_keepdim():
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float32, device="cpu")

    ref = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
    tri = vector_norm(x, ord=2, dim=[1], keepdim=True)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


def test_vector_norm_inf():
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32, device="cpu")

    ref = torch.linalg.vector_norm(x, ord=float("inf"))
    tri = vector_norm(x, ord=float("inf"))

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)
