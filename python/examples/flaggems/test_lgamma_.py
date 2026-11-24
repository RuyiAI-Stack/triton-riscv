import pytest
import torch

from .lgamma_ import lgamma, lgamma_


def test_lgamma():
    x = torch.linspace(0.5, 5.0, 1024, dtype=torch.float32, device="cpu")

    out = lgamma(x)
    ref = torch.lgamma(x)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_lgamma_inplace():
    x = torch.tensor([0.5, 1.0, 3.0], dtype=torch.float32, device="cpu")
    ref = torch.lgamma(x)

    lgamma_(x)

    torch.testing.assert_close(x, ref, rtol=1e-4, atol=1e-4)


def test_lgamma_negative_noninteger_inputs():
    x = torch.tensor([-3.5, -1.5, -0.5, 0.5], dtype=torch.float64, device="cpu")

    out = lgamma(x)
    ref = torch.lgamma(x)

    torch.testing.assert_close(out, ref, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lgamma_poles_and_infinity(dtype):
    x = torch.tensor(
        [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, float("inf")],
        dtype=dtype,
        device="cpu",
    )

    torch.testing.assert_close(lgamma(x), torch.lgamma(x), equal_nan=True)
