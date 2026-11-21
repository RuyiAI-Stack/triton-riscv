import importlib

import pytest
import torch

from .view_as_real_copy import view_as_real_copy
from .view_as_complex_copy import view_as_complex_copy


@pytest.mark.parametrize(
    "cdt,rdt", [(torch.complex64, torch.float32), (torch.complex128, torch.float64)]
)
def test_view_as_real_copy_matches_aten(cdt, rdt):
    torch.manual_seed(0)
    x = torch.randn(3, 4, dtype=cdt)
    out = view_as_real_copy(x)
    ref = torch.ops.aten.view_as_real_copy(x)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.dtype == rdt and out.shape == (3, 4, 2)


def test_view_as_real_copy_strided_input():
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=torch.complex128).t()  # (6, 4) non-contiguous
    out = view_as_real_copy(x)
    ref = torch.ops.aten.view_as_real_copy(x)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_view_as_real_copy_higher_rank_strided():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, dtype=torch.complex64).permute(1, 0, 2)
    out = view_as_real_copy(x)
    ref = torch.ops.aten.view_as_real_copy(x)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_view_as_real_copy_non_complex_raises():
    x = torch.randn(3)
    with pytest.raises(RuntimeError, match="only supported for complex"):
        view_as_real_copy(x)


def test_view_as_real_copy_empty():
    x = torch.empty(0, 3, dtype=torch.complex64)
    out = view_as_real_copy(x)
    assert out.shape == (0, 3, 2)


def test_view_as_real_copy_is_value_copy():
    torch.manual_seed(0)
    x = torch.randn(4, dtype=torch.complex64)
    out = view_as_real_copy(x)
    out[0, 0] = 99.0
    assert x[0].real != 99.0


def test_view_as_real_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.view_as_real_copy")
    launches = []
    real_run = mod._view_as_real_copy_contig_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._view_as_real_copy_contig_kernel, "run", spy_run)
    torch.manual_seed(0)
    x = torch.randn(3, 5, dtype=torch.complex128)
    out = view_as_real_copy(x)
    assert len(launches) >= 1
    assert out.shape == (3, 5, 2)
    torch.testing.assert_close(out[:, :, 0], x.real, rtol=0, atol=0)
    torch.testing.assert_close(out[:, :, 1], x.imag, rtol=0, atol=0)


@pytest.mark.parametrize(
    "rdt,cdt", [(torch.float32, torch.complex64), (torch.float64, torch.complex128)]
)
def test_view_as_complex_copy_matches_aten(rdt, cdt):
    torch.manual_seed(0)
    x = torch.randn(3, 4, 2, dtype=rdt)
    out = view_as_complex_copy(x)
    ref = torch.ops.aten.view_as_complex_copy(x)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.dtype == cdt and out.shape == (3, 4)


def test_view_as_complex_copy_strided_input():
    torch.manual_seed(0)
    base = torch.randn(4, 6, 2, dtype=torch.float64)
    x = base[::2]  # strided rows, pair dim keeps stride 1
    assert not x.is_contiguous()
    out = view_as_complex_copy(x)
    ref = torch.ops.aten.view_as_complex_copy(x)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_view_as_complex_copy_roundtrip():
    torch.manual_seed(0)
    z = torch.randn(3, 4, dtype=torch.complex128)
    back = view_as_complex_copy(view_as_real_copy(z))
    torch.testing.assert_close(back, z, rtol=0, atol=0)


def test_view_as_complex_copy_wrong_last_dim_raises():
    x = torch.randn(3, 3)
    with pytest.raises(RuntimeError, match="last dimension of size 2"):
        view_as_complex_copy(x)


def test_view_as_complex_copy_wrong_dtype_raises():
    x = torch.randn(3, 2, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="half, float and double"):
        view_as_complex_copy(x)


def test_view_as_complex_copy_zero_dim_raises():
    x = torch.tensor(1.0)
    with pytest.raises(RuntimeError, match="one or more dimensions"):
        view_as_complex_copy(x)


def test_view_as_complex_copy_empty():
    x = torch.empty(0, 3, 2, dtype=torch.float64)
    out = view_as_complex_copy(x)
    assert out.shape == (0, 3)


def test_view_as_complex_copy_triton_path(monkeypatch):
    vcmod = importlib.import_module("flaggems.view_as_complex_copy")

    launches = []
    real_run = vcmod._view_as_complex_copy_contig_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(vcmod._view_as_complex_copy_contig_kernel, "run", spy_run)
    torch.manual_seed(0)
    x = torch.randn(3, 5, 2, dtype=torch.float64)
    out = view_as_complex_copy(x)
    assert len(launches) >= 1
    torch.testing.assert_close(out.real, x[..., 0], rtol=0, atol=0)
    torch.testing.assert_close(out.imag, x[..., 1], rtol=0, atol=0)
