import importlib

import pytest
import torch

from ._conj import _conj

COMPLEX_DTYPES = [torch.complex64, torch.complex128]
# torch._conj raises an internal RuntimeError for every non-complex dtype on
# CPU (isComplexType assert); the port keeps the upstream FlagGems contract of
# an explicit RuntimeError for the same inputs.
NON_COMPLEX_DTYPES = [torch.float32, torch.float64, torch.int32, torch.bool]


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,), (4, 128)])
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test__conj(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    ref = torch._conj(x)
    tri = _conj(x)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri.dtype == dtype
    # The port materializes a fresh contiguous tensor with no conjugate bit.
    assert not tri.is_conj()
    assert tri.is_contiguous()
    assert tri.data_ptr() != x.data_ptr()


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test__conj_pure_real(dtype):
    # All-zero imaginary parts: conj is a sign flip of +/-0.0.
    torch.manual_seed(0)
    x = torch.complex(torch.randn(256), torch.zeros(256)).to(dtype)
    torch.testing.assert_close(_conj(x), torch._conj(x), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test__conj_pure_imag(dtype):
    torch.manual_seed(0)
    x = torch.complex(torch.zeros(256), torch.randn(256)).to(dtype)
    torch.testing.assert_close(_conj(x), torch._conj(x), rtol=0, atol=0)


def test__conj_nan_imag():
    torch.manual_seed(0)
    x = torch.complex(torch.randn(64), torch.full((64,), float("nan")))
    tri = _conj(x)
    ref = torch.conj_physical(x)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test__conj_conj_bit(dtype):
    # Input carrying the conjugate bit: _conj must return the physical data of
    # the original (unconjugated) tensor, exactly like aten's bit-cleared alias.
    torch.manual_seed(0)
    x = torch.randn(64, dtype=dtype)
    xc = x.conj()
    assert xc.is_conj()
    tri = _conj(xc)
    ref = torch._conj(xc)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    torch.testing.assert_close(tri, x, rtol=0, atol=0)
    assert not tri.is_conj()
    assert tri.data_ptr() != xc.data_ptr()


@pytest.mark.parametrize("dtype", NON_COMPLEX_DTYPES)
def test__conj_non_complex_raises(dtype):
    x = torch.zeros(8, dtype=dtype)
    with pytest.raises(RuntimeError):
        _conj(x)


@pytest.mark.parametrize("shape", [(0,), (0, 37)])
def test__conj_empty(shape):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=torch.complex64)
    tri = _conj(x)
    torch.testing.assert_close(tri, torch._conj(x), rtol=0, atol=0)
    assert tri.shape == x.shape


def test__conj_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(8, 6, dtype=torch.complex64).t()
    assert not x.is_contiguous()
    tri = _conj(x)
    torch.testing.assert_close(tri, torch._conj(x), rtol=0, atol=0)
    assert tri.shape == x.shape
    # The port returns a materialized contiguous copy, unlike the lazy view
    # torch._conj keeps for non-contiguous inputs.
    assert tri.is_contiguous()


def test__conj_triton_path(monkeypatch):
    # Smoke: the complex64 main path must launch the Triton kernel.
    mod = importlib.import_module("flaggems._conj")

    launches = []
    real_run = mod._conj_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._conj_kernel, "run", spy_run)
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.complex64)
    out = mod._conj(x)
    assert len(launches) == 1, "_conj(x) must go through the Triton kernel"
    torch.testing.assert_close(out, torch._conj(x), rtol=0, atol=0)
