import importlib

import pytest
import torch

from .chalf import chalf

# Real dtypes exercised by the upstream chalf tests (float16/32/64) plus the
# integer input the upstream kernel comment explicitly claims; every one of
# these Tensor.chalf() conversions is legal on this CPU torch build (probed).
REAL_DTYPES = [torch.float16, torch.float32, torch.float64, torch.int32]
COMPLEX_DTYPES = [torch.complex64, torch.complex128]


def _real_input(shape, dtype):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=torch.float32).to(dtype)
    return torch.randint(-1000, 1000, shape, dtype=dtype)


# ComplexHalf has no equal/assert support for direct logical comparison on
# this build ("equal_cpu" not implemented for ComplexHalf), so every check
# compares the view_as_real float16 views, mirroring the upstream tests.
def _as_real(t):
    return torch.view_as_real(t)


@pytest.mark.parametrize("shape", [(512,), (33,), (4, 128), (2, 3, 4)])
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_chalf_real(shape, dtype):
    torch.manual_seed(0)
    x = _real_input(shape, dtype)
    ref = x.chalf()
    tri = chalf(x)
    assert tri.dtype == torch.complex32
    torch.testing.assert_close(_as_real(tri), _as_real(ref), rtol=0, atol=0)
    assert tri.shape == x.shape


@pytest.mark.parametrize("shape", [(256,), (513,), (32, 64)])
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_chalf_complex(shape, dtype):
    torch.manual_seed(0)
    real = torch.randn(shape, dtype=torch.float64)
    imag = torch.randn(shape, dtype=torch.float64)
    x = torch.complex(real, imag).to(dtype)
    ref = x.chalf()
    tri = chalf(x)
    assert tri.dtype == torch.complex32
    torch.testing.assert_close(_as_real(tri), _as_real(ref), rtol=0, atol=0)


def test_chalf_complex_nan_inf():
    v = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0, 1.5],
    )
    x = torch.complex(v, v.flip(0))
    tri = chalf(x)
    ref = x.chalf()
    assert tri.dtype == torch.complex32
    torch.testing.assert_close(
        _as_real(tri), _as_real(ref), rtol=0, atol=0, equal_nan=True
    )


def test_chalf_complex_rounding_boundaries():
    # Inexact and saturating values through the float64 -> float16 cast:
    # 0.1/1-3 round to nearest, 1e300 overflows to inf.
    v = torch.tensor(
        [0.1, 1.0 / 3.0, 3.14159265358979, 1e300, 65504.0],
        dtype=torch.float64,
    )
    x = torch.complex(v, -v)
    tri = chalf(x)
    ref = x.chalf()
    torch.testing.assert_close(_as_real(tri), _as_real(ref), rtol=0, atol=0)


def test_chalf_conj_bit():
    # A conjugate-bit input must materialize its logical (conjugated) values,
    # matching what torch's conversion produces.
    torch.manual_seed(0)
    original = torch.randn(64, dtype=torch.complex64)
    x = original.conj()
    assert x.is_conj()
    tri = chalf(x)
    ref = x.chalf()
    assert tri.dtype == torch.complex32
    torch.testing.assert_close(_as_real(tri), _as_real(ref), rtol=0, atol=0)
    # Genuinely the conjugated (logical) values, not the raw a+bi storage.
    expected = torch.view_as_real(original.to(torch.complex32)).clone()
    expected[:, 1] = -expected[:, 1]
    torch.testing.assert_close(_as_real(tri), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0,), (0, 7), (2, 0, 4)])
def test_chalf_empty(shape):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=torch.float32)
    tri = chalf(x)
    ref = x.chalf()
    assert tri.dtype == torch.complex32
    assert tri.shape == x.shape
    torch.testing.assert_close(_as_real(tri), _as_real(ref), rtol=0, atol=0)


def test_chalf_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(8, 6).t()
    assert not x.is_contiguous()
    tri = chalf(x)
    ref = x.chalf()
    torch.testing.assert_close(_as_real(tri), _as_real(ref), rtol=0, atol=0)


def test_chalf_memory_format_kwarg():
    # The memory_format parameter is accepted (upstream signature keeps it but
    # does not act on it); both call forms must produce the same conversion.
    torch.manual_seed(0)
    x = torch.randn(64)
    tri_kw = chalf(x, memory_format=torch.preserve_format)
    tri_pos = chalf(x, torch.contiguous_format)
    ref = x.chalf()
    torch.testing.assert_close(_as_real(tri_kw), _as_real(ref), rtol=0, atol=0)
    torch.testing.assert_close(_as_real(tri_pos), _as_real(ref), rtol=0, atol=0)


def test_chalf_real_triton_path(monkeypatch):
    # Smoke: the real-input path must launch the Triton kernel.
    mod = importlib.import_module("flaggems.chalf")

    launches = []
    real_run = mod.chalf_real_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.chalf_real_kernel, "run", spy_run)
    torch.manual_seed(0)
    x = torch.randn(1024)
    out = mod.chalf(x)
    assert len(launches) == 1, "chalf(real) must go through the Triton kernel"
    torch.testing.assert_close(
        torch.view_as_real(out), torch.view_as_real(x.chalf()), rtol=0, atol=0
    )


def test_chalf_complex_triton_path(monkeypatch):
    # Smoke: the complex-input path must launch the Triton kernel.
    mod = importlib.import_module("flaggems.chalf")

    launches = []
    real_run = mod.chalf_complex_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.chalf_complex_kernel, "run", spy_run)
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.complex64)
    out = mod.chalf(x)
    assert len(launches) == 1, "chalf(complex) must go through the Triton kernel"
    torch.testing.assert_close(
        torch.view_as_real(out), torch.view_as_real(x.chalf()), rtol=0, atol=0
    )
