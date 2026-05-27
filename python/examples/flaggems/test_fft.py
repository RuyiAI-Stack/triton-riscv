import pytest
import torch
import torch.fft

from .fft import fft


@pytest.mark.parametrize("M, N", [(2, 8), (3, 16)])
def test_fft(M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float32)

    ref = torch.fft.fft(x)
    tri = fft(x)

    torch.testing.assert_close(tri.real, ref.real, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(tri.imag, ref.imag, rtol=1e-3, atol=1e-3)
