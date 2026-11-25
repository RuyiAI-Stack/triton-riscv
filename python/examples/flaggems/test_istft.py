import torch

from .istft import istft


def test_istft():
    torch.manual_seed(0)
    signal = torch.randn(32)
    window = torch.hann_window(16)
    spectrum = torch.stft(
        signal,
        16,
        hop_length=8,
        win_length=16,
        window=window,
        return_complex=True,
        center=True,
    )
    rebuilt = istft(
        spectrum, 16, hop_length=8, win_length=16, window=window, center=True, length=32
    )
    torch.testing.assert_close(
        rebuilt,
        torch.istft(
            spectrum,
            16,
            hop_length=8,
            win_length=16,
            window=window,
            center=True,
            length=32,
        ),
        rtol=1e-4,
        atol=1e-3,
    )
