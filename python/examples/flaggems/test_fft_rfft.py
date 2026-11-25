import torch

from .fft_rfft import fft_rfft


def test_fft_rfft():
    torch.manual_seed(0)
    signal = torch.randn(16)
    torch.testing.assert_close(
        fft_rfft(signal), torch.fft.rfft(signal), rtol=1e-4, atol=1e-4
    )
