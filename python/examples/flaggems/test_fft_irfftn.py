import torch

from .fft_irfftn import fft_irfftn


def test_fft_irfftn():
    torch.manual_seed(0)
    signal = torch.randn(16)
    spectrum = torch.fft.rfft(signal)
    torch.testing.assert_close(
        fft_irfftn(spectrum, s=(16,), dim=-1),
        torch.fft.irfft(spectrum, n=16),
        rtol=1e-4,
        atol=1e-4,
    )
