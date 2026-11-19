import pytest
import torch

from .std_mean import std_mean


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_std_mean_global(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 3.0 + 5.0

    ref_std, ref_mean = torch.std_mean(x)
    tri_std, tri_mean = std_mean(x)

    torch.testing.assert_close(tri_std, ref_std, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_mean, ref_mean, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, dim", [((16, 256), 1), ((16, 256), 0), ((4, 8, 32), 2)]
)
def test_std_mean_dim(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32) * 3.0 + 5.0

    ref_std, ref_mean = torch.std_mean(x, dim=dim)
    tri_std, tri_mean = std_mean(x, dim=dim)

    torch.testing.assert_close(tri_std, ref_std, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_mean, ref_mean, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("correction", [0, 1, 2])
def test_std_mean_correction(correction):
    torch.manual_seed(0)
    x = torch.randn(16, 256, device="cpu", dtype=torch.float32) * 3.0

    ref_std, ref_mean = torch.std_mean(x, dim=1, correction=correction)
    tri_std, tri_mean = std_mean(x, dim=1, correction=correction)

    torch.testing.assert_close(tri_std, ref_std, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_mean, ref_mean, rtol=1e-4, atol=1e-4)


def test_std_mean_keepdim():
    torch.manual_seed(0)
    x = torch.randn(16, 256, device="cpu", dtype=torch.float32) * 3.0

    ref_std, ref_mean = torch.std_mean(x, dim=1, keepdim=True)
    tri_std, tri_mean = std_mean(x, dim=1, keepdim=True)

    assert tri_std.shape == ref_std.shape
    assert tri_mean.shape == ref_mean.shape
    torch.testing.assert_close(tri_std, ref_std, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_mean, ref_mean, rtol=1e-4, atol=1e-4)


def test_std_mean_large_offset():
    # A large mean with small variance exposes one-pass sum-of-squares
    # cancellation; the two-pass kernel must stay accurate.
    torch.manual_seed(0)
    x = torch.randn(16, 256, device="cpu", dtype=torch.float32) * 1e-3 + 1e4

    ref_std, ref_mean = torch.std_mean(x, dim=1)
    tri_std, tri_mean = std_mean(x, dim=1)

    torch.testing.assert_close(tri_std, ref_std, rtol=1e-3, atol=1e-6)
    torch.testing.assert_close(tri_mean, ref_mean, rtol=1e-4, atol=1e-4)


def test_std_mean_fp64_fallback():
    torch.manual_seed(0)
    x = torch.randn(1024, device="cpu", dtype=torch.float64) * 3.0 + 5.0

    ref_std, ref_mean = torch.std_mean(x)
    tri_std, tri_mean = std_mean(x)

    torch.testing.assert_close(tri_std, ref_std)
    torch.testing.assert_close(tri_mean, ref_mean)


def test_std_mean_bf16_fallback():
    # bf16 delegates to torch; this test pins that fallback.
    torch.manual_seed(0)
    x = (torch.randn(128, dtype=torch.float32) * 3.0).to(torch.bfloat16)

    ref_std, ref_mean = torch.std_mean(x)
    tri_std, tri_mean = std_mean(x)

    torch.testing.assert_close(tri_std, ref_std, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(tri_mean, ref_mean, rtol=1e-2, atol=1e-2)


def test_std_mean_global_keepdim():
    torch.manual_seed(0)
    x = torch.randn(8, 32, device="cpu", dtype=torch.float32) * 3.0

    ref_std, ref_mean = torch.std_mean(x, keepdim=True)
    tri_std, tri_mean = std_mean(x, keepdim=True)

    assert tri_std.shape == ref_std.shape
    assert tri_mean.shape == ref_mean.shape
    torch.testing.assert_close(tri_std, ref_std, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_mean, ref_mean, rtol=1e-4, atol=1e-4)


def test_std_mean_fp16_fallback():
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float16)
    tri_s, tri_m = std_mean(x)  # torch order: (std, mean)
    ref_s, ref_m = torch.std_mean(x)
    torch.testing.assert_close(tri_m, ref_m, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(tri_s, ref_s, rtol=1e-2, atol=1e-2)


def test_std_mean_empty_and_zero_dim():
    torch.testing.assert_close(
        std_mean(torch.empty(0)), torch.std_mean(torch.empty(0)), equal_nan=True
    )
    s, m = std_mean(torch.empty(3, 0), dim=1)
    rs, rm = torch.std_mean(torch.empty(3, 0), dim=1)
    torch.testing.assert_close(s, rs, equal_nan=True)
    torch.testing.assert_close(m, rm, equal_nan=True)
