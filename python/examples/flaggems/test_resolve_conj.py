import pytest
import torch

from .resolve_conj import resolve_conj


def test_resolve_conj_real_tensor():
    x = torch.randn(4, 4)
    r = resolve_conj(x)
    torch.testing.assert_close(r, x)


@pytest.mark.parametrize(
    "shape",
    [(4, 4), (16, 256), (1024,), (2, 4, 8), (100, 200, 5)],
)
def test_resolve_conj_complex64(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.complex64)
    y = x.conj()
    assert y.is_conj()
    r = resolve_conj(y)
    ref = torch.resolve_conj(y)
    torch.testing.assert_close(r, ref)


@pytest.mark.parametrize(
    "shape",
    [(4, 4), (16, 256), (1024,), (2, 4, 8)],
)
def test_resolve_conj_not_conj(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.complex64)
    assert not x.is_conj()
    r = resolve_conj(x)
    ref = torch.resolve_conj(x)
    torch.testing.assert_close(r, ref)


@pytest.mark.parametrize(
    "shape",
    [(4, 4), (16, 256), (1024,)],
)
def test_resolve_conj_large_2d(shape):
    torch.manual_seed(0)
    x = torch.randn(100, 20000, dtype=torch.complex64)
    y = x.conj()
    assert y.is_conj()
    r = resolve_conj(y)
    ref = torch.resolve_conj(y)
    torch.testing.assert_close(r, ref)
