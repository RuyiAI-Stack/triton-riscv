import pytest
import torch

from .nansum import nansum


def _make_nan_input(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32) * 3.0
    x[0::7] = float("nan")
    return x


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_nansum_global(size):
    x = _make_nan_input((size,))

    ref = torch.nansum(x)
    tri = nansum(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, dim", [((16, 256), 1), ((16, 256), 0), ((4, 8, 32), 2)]
)
def test_nansum_dim(shape, dim):
    x = _make_nan_input(shape)

    ref = torch.nansum(x, dim=dim)
    tri = nansum(x, dim=dim)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_nansum_all_nan_column():
    # A fully-NaN reduction lane must sum to 0, matching torch.
    x = torch.randn(8, 32, device="cpu", dtype=torch.float32)
    x[:, 0] = float("nan")

    ref = torch.nansum(x, dim=0)
    tri = nansum(x, dim=0)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_nansum_keepdim():
    x = _make_nan_input((16, 256))

    ref = torch.nansum(x, dim=1, keepdim=True)
    tri = nansum(x, dim=1, keepdim=True)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16])
def test_nansum_fp16(dtype):
    x = (_make_nan_input((128,))).to(dtype)

    ref = torch.nansum(x)
    tri = nansum(x)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@BF16_XFAIL
def test_nansum_bf16():
    torch.manual_seed(0)
    x = (torch.randn(128, dtype=torch.float32) * 3.0).to(torch.bfloat16)
    x[0::7] = float("nan")

    ref = torch.nansum(x)
    tri = nansum(x)

    torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)


def test_nansum_fp64_fallback():
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float64) * 3.0
    x[0::7] = float("nan")

    ref = torch.nansum(x)
    tri = nansum(x)

    torch.testing.assert_close(tri, ref)


def test_nansum_zero_length_dim():
    torch.testing.assert_close(
        nansum(torch.empty(3, 0), dim=1), torch.nansum(torch.empty(3, 0), dim=1)
    )
