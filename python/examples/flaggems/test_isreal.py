import pytest
import torch

from .isreal import isreal


def _make_complex(shape, dtype):
    """Build a complex tensor whose imaginary part is zero on ~half of it."""
    float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(shape, dtype=float_dtype)
    imag = torch.randn(shape, dtype=float_dtype)
    if imag.dim() > 0:
        imag[::2] = 0
    else:
        imag = torch.zeros_like(imag)
    return torch.complex(real, imag).to(dtype)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool],
)
def test_isreal_real(dtype, size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32).to(dtype)

    ref = torch.isreal(x)
    out = isreal(x)

    assert out.dtype == torch.bool
    assert out.shape == x.shape
    assert torch.equal(out, ref)
    assert torch.all(out)


@pytest.mark.parametrize(
    "shape",
    [(512,), (1023,), (1024,), (16, 64), (4, 8, 16), (2, 4, 8, 16)],
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_isreal_complex(dtype, shape):
    torch.manual_seed(0)
    x = _make_complex(shape, dtype)

    ref = torch.isreal(x)
    out = isreal(x)

    assert out.dtype == torch.bool
    assert out.shape == x.shape
    assert torch.equal(out, ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.complex128])
def test_isreal_special_values(dtype):
    """Zero, negative zero, NaN and infinity, including a NaN imaginary part."""
    if dtype.is_complex:
        float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        real = [0.0, -0.0, 1.0, -1.0, float("nan"), float("inf"), 1.0, 1.0, -0.0]
        imag = [0.0, -0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), float("inf"), -0.0]
        x = torch.complex(
            torch.tensor(real, dtype=float_dtype),
            torch.tensor(imag, dtype=float_dtype),
        ).to(dtype)
    else:
        x = torch.tensor(
            [0.0, -0.0, 1.0, -1.0, float("nan"), float("inf"), 1.0, 1.0, -0.0],
            dtype=dtype,
        )

    ref = torch.isreal(x)
    out = isreal(x)

    assert out.dtype == torch.bool
    assert torch.equal(out, ref)
    # -0.0 imaginary part still counts as real (IEEE signed zeros compare equal).
    assert out[-1].item() is True


def test_isreal_scalar():
    x = torch.tensor(1.0 + 0j)
    assert isreal(x).dtype == torch.bool
    assert isreal(x).item() is True
    assert isreal(torch.tensor(3.0)).item() is True
    assert isreal(torch.tensor(1.0 + 2.0j)).item() is False


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.complex128])
def test_isreal_non_contiguous(dtype):
    torch.manual_seed(0)
    if dtype.is_complex:
        base = _make_complex((32, 48), dtype)
    else:
        base = torch.randn((32, 48), dtype=dtype)
    x = base.t()
    assert not x.is_contiguous()

    ref = torch.isreal(x)
    out = isreal(x)

    assert out.dtype == torch.bool
    assert out.shape == x.shape
    assert torch.equal(out, ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_isreal_empty(dtype):
    x = torch.empty(0, dtype=dtype)

    out = isreal(x)

    assert out.dtype == torch.bool
    assert out.numel() == 0
    assert torch.equal(out, torch.isreal(x))


def test_isreal_conj_bit():
    """A lazy conjugate only flips the sign of the imaginary part."""
    torch.manual_seed(0)
    base = _make_complex((1024,), torch.complex64)
    x = base.conj()
    assert x.is_conj()

    out = isreal(x)

    assert out.dtype == torch.bool
    assert torch.equal(out, torch.isreal(x))


def test_isreal_independent_storage():
    """isreal writes a freshly allocated bool tensor (copy semantics)."""
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32)

    out = isreal(x)

    assert out.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()


def test_isreal_rejects_non_tensor():
    with pytest.raises(TypeError):
        isreal(3.14)


def test_isreal_rejects_unsupported_dtype():
    x = torch.tensor([1.0], dtype=torch.float8_e4m3fn)
    with pytest.raises(TypeError):
        isreal(x)
