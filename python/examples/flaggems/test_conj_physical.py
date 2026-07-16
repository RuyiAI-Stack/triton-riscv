import pytest
import torch

from .conj_physical import conj_physical


@pytest.mark.parametrize("shape", [(4, 4), (16, 16)])
def test_conj_physical(shape):
    x = torch.randn(shape, dtype=torch.complex64, device="cpu")
    ref = torch.conj_physical(x)
    tri = conj_physical(x)
    torch.testing.assert_close(tri, ref)


def test_conj_physical_real():
    x = torch.randn(4, 4, dtype=torch.float32, device="cpu")
    r = conj_physical(x)
    assert r is x
