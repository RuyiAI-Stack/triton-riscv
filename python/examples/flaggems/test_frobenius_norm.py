import pytest
import torch

from .frobenius_norm import frobenius_norm


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_frobenius_norm_global(size):
    torch.manual_seed(0)
    x = (
        torch.randn(size, size, device="cpu", dtype=torch.float32)
        if size < 600
        else torch.randn(16, size, device="cpu", dtype=torch.float32)
    )

    ref = torch.linalg.norm(x, ord="fro")
    tri = frobenius_norm(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_frobenius_norm_dim():
    torch.manual_seed(0)
    x = torch.randn(8, 32, device="cpu", dtype=torch.float32) * 3.0

    ref = torch.linalg.norm(x, ord="fro", dim=(0, 1))
    tri = frobenius_norm(x, dim=(0, 1))

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_frobenius_norm_keepdim():
    torch.manual_seed(0)
    x = torch.randn(8, 32, device="cpu", dtype=torch.float32) * 3.0

    ref = torch.linalg.norm(x, ord="fro", dim=(0, 1), keepdim=True)
    tri = frobenius_norm(x, dim=(0, 1), keepdim=True)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_frobenius_norm_3d_rows():
    # Rows of a 3D tensor: frobenius over the last two dims.
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32, device="cpu", dtype=torch.float32) * 3.0

    ref = torch.linalg.norm(x, ord="fro", dim=(1, 2))
    tri = frobenius_norm(x, dim=(1, 2))

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_frobenius_norm_fp64_fallback():
    torch.manual_seed(0)
    x = torch.randn(16, 64, device="cpu", dtype=torch.float64) * 3.0

    ref = torch.linalg.norm(x, ord="fro")
    tri = frobenius_norm(x)

    torch.testing.assert_close(tri, ref)


def test_frobenius_norm_bf16_fallback():
    # bf16 delegates to torch; this test pins that fallback.
    torch.manual_seed(0)
    x = (torch.randn(8, 32, dtype=torch.float32) * 3.0).to(torch.bfloat16)

    ref = torch.linalg.norm(x.float(), ord="fro").to(torch.bfloat16)
    tri = frobenius_norm(x)

    torch.testing.assert_close(tri.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_frobenius_norm_fp16_fallback():
    torch.manual_seed(0)
    x = torch.randn(16, 17, dtype=torch.float16)
    torch.testing.assert_close(
        frobenius_norm(x), torch.linalg.norm(x), rtol=1e-2, atol=1e-2
    )


def test_frobenius_norm_zero_length_dim():
    torch.testing.assert_close(
        frobenius_norm(torch.empty(3, 0), dim=1),
        torch.linalg.vector_norm(torch.empty(3, 0), dim=1),
    )
