import pytest
import torch

from .log1p import log1p, log1p_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_log1p(shape):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device="cpu") + 0.1

    ref = torch.log1p(x)
    out = log1p(x)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_log1p_out(shape):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device="cpu") + 0.1
    out = torch.empty_like(x)

    ref = torch.log1p(x)
    returned = log1p_out(x, out=out)

    assert returned is out
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_log1p_small_float64_stability():
    x = torch.tensor([1e-16, -1e-16, 1e-12], dtype=torch.float64, device="cpu")

    out = log1p(x)
    ref = torch.log1p(x)

    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log1p_special_values(dtype):
    x = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), -1.0, 0.0],
        dtype=dtype,
        device="cpu",
    )

    torch.testing.assert_close(log1p(x), torch.log1p(x), equal_nan=True)


def test_log1p_out_resizes_output():
    x = torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32, device="cpu")
    out = torch.empty(0, dtype=torch.float32, device="cpu")

    returned = log1p_out(x, out=out)

    assert returned is out
    assert out.shape == x.shape
    torch.testing.assert_close(out, torch.log1p(x))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_log1p_complex(dtype):
    x = torch.tensor([0.5 + 0.25j, -0.2 + 0.1j], dtype=dtype, device="cpu")

    out = log1p(x)
    ref = torch.log1p(x)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("out_dtype", "should_raise"),
    [
        (torch.float64, False),
        (torch.float16, False),
        (torch.complex64, False),
        (torch.int32, True),
    ],
)
def test_log1p_out_dtype_contract(out_dtype, should_raise):
    x = torch.tensor([0.25, 0.5], dtype=torch.float32, device="cpu")
    out = torch.empty(x.shape, dtype=out_dtype, device="cpu")

    if should_raise:
        with pytest.raises(RuntimeError, match="can't be cast"):
            log1p_out(x, out=out)
        return

    returned = log1p_out(x, out=out)
    assert returned is out
    torch.testing.assert_close(out, torch.log1p(x).to(out_dtype), rtol=1e-3, atol=1e-3)


def test_log1p_out_complex_widening():
    x = torch.tensor([0.5 + 0.25j], dtype=torch.complex64, device="cpu")
    out = torch.empty_like(x, dtype=torch.complex128)

    returned = log1p_out(x, out=out)

    assert returned is out
    torch.testing.assert_close(out, torch.log1p(x).to(out.dtype), rtol=1e-5, atol=1e-5)
