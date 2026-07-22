import pytest
import torch

from .erfinv_ import erfinv, erfinv_


def test_erfinv():
    x = torch.linspace(-0.8, 0.8, 1024, dtype=torch.float32, device="cpu")

    out = erfinv(x)
    ref = torch.erfinv(x)

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_erfinv_inplace():
    x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32, device="cpu")
    ref = torch.erfinv(x)

    erfinv_(x)

    torch.testing.assert_close(x, ref, rtol=2e-2, atol=2e-2)


def test_erfinv_preserves_float64_precision():
    x = torch.tensor([-0.75, -0.25, 0.25, 0.75], dtype=torch.float64, device="cpu")

    out = erfinv(x)
    ref = torch.erfinv(x)

    torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)


def test_erfinv_float64_near_one():
    x = torch.tensor(
        [-0.999999, -0.999, 0.999, 0.999999], dtype=torch.float64, device="cpu"
    )

    out = erfinv(x)
    ref = torch.erfinv(x)

    torch.testing.assert_close(out, ref, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_erfinv_special_values(dtype):
    x = torch.tensor(
        [-1.1, -1.0, 0.0, 1.0, 1.1, float("inf"), float("-inf"), float("nan")],
        dtype=dtype,
        device="cpu",
    )

    torch.testing.assert_close(
        erfinv(x), torch.erfinv(x), rtol=2e-2, atol=2e-2, equal_nan=True
    )
