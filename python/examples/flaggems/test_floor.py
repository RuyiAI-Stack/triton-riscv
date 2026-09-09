import pytest
import torch

from .floor import floor, floor_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_floor(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu") * 10.0

    ref = torch.floor(x)
    out = floor(x)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_floor_out(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu") * 10.0
    out = torch.empty_like(x)

    ref = torch.floor(x)
    returned = floor_out(x, out=out)

    assert returned is out
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_floor_preserves_float64_precision():
    x = torch.tensor(
        [float(2**24) - 0.25, -float(2**24) + 0.75],
        dtype=torch.float64,
        device="cpu",
    )

    out = floor(x)
    ref = torch.floor(x)

    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


def test_floor_out_resizes_output():
    x = torch.tensor([1.2, -3.4, 5.6], dtype=torch.float32, device="cpu")
    out = torch.empty(0, dtype=torch.float32, device="cpu")

    returned = floor_out(x, out=out)

    assert returned is out
    assert out.shape == x.shape
    torch.testing.assert_close(out, torch.floor(x))


def test_floor_out_rejects_dtype_mismatch():
    x = torch.tensor([1.2, -3.4], dtype=torch.float32, device="cpu")
    out = torch.empty_like(x, dtype=torch.float64)

    with pytest.raises(RuntimeError, match="expected Float"):
        floor_out(x, out=out)


def test_floor_integer_tensor_is_unchanged():
    x = torch.tensor([-4, 0, 11], dtype=torch.int32, device="cpu")

    out = floor(x)

    assert out.dtype == x.dtype
    torch.testing.assert_close(out, x)


def test_floor_out_rejects_mismatched_dtype():
    x = torch.tensor([1.2], dtype=torch.float32, device="cpu")
    out = torch.empty(1, dtype=torch.int32, device="cpu")

    with pytest.raises(RuntimeError):
        floor_out(x, out=out)
