import pytest
import torch

from .nanmean import nanmean


def _make_nan_input(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32) * 3.0
    x[0::7] = float("nan")
    return x


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_nanmean_global(size):
    x = _make_nan_input((size,))

    ref = torch.nanmean(x)
    tri = nanmean(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4, equal_nan=True)


@pytest.mark.parametrize(
    "shape, dim", [((16, 256), 1), ((16, 256), 0), ((4, 8, 32), 2)]
)
def test_nanmean_dim(shape, dim):
    x = _make_nan_input(shape)

    ref = torch.nanmean(x, dim=dim)
    tri = nanmean(x, dim=dim)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_nanmean_all_nan_column():
    # A fully-NaN reduction lane must produce NaN, matching torch.
    x = torch.randn(8, 32, device="cpu", dtype=torch.float32)
    x[:, 0] = float("nan")

    ref = torch.nanmean(x, dim=0)
    tri = nanmean(x, dim=0)

    torch.testing.assert_close(tri, ref, equal_nan=True)


def test_nanmean_keepdim():
    x = _make_nan_input((16, 256))

    ref = torch.nanmean(x, dim=1, keepdim=True)
    tri = nanmean(x, dim=1, keepdim=True)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4, equal_nan=True)


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@BF16_XFAIL
def test_nanmean_bf16():
    torch.manual_seed(0)
    x = (torch.randn(128, dtype=torch.float32) * 3.0).to(torch.bfloat16)
    x[0::7] = float("nan")

    ref = torch.nanmean(x)
    tri = nanmean(x)

    torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)


def test_nanmean_fp64_fallback():
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float64) * 3.0
    x[0::7] = float("nan")

    ref = torch.nanmean(x)
    tri = nanmean(x)

    torch.testing.assert_close(tri, ref, equal_nan=True)


def test_nanmean_global_small_nonzero_mean():
    # M=17 leaves 7 padding lanes in the last block; a nonzero mean makes
    # any count leakage visible (the old bug returned 153/24 = 6.375).
    x = torch.arange(1, 18, dtype=torch.float32)
    x[3] = float("nan")

    ref = torch.nanmean(x)
    tri = nanmean(x)

    torch.testing.assert_close(tri, ref)


def test_nanmean_global_all_nan():
    x = torch.full((37,), float("nan"))

    ref = torch.nanmean(x)
    tri = nanmean(x)

    torch.testing.assert_close(tri, ref, equal_nan=True)


def test_nanmean_fp16():
    torch.manual_seed(0)
    x = torch.randn(1023, dtype=torch.float16)
    x[::7] = float("nan")
    torch.testing.assert_close(nanmean(x), torch.nanmean(x), rtol=1e-2, atol=1e-2)


def test_nanmean_zero_length_dim():
    torch.testing.assert_close(
        nanmean(torch.empty(3, 0), dim=1),
        torch.nanmean(torch.empty(3, 0), dim=1),
        equal_nan=True,
    )
