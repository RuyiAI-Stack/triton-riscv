import torch

from .fft_ifftshift import fft_ifftshift


def test_fft_ifftshift():
    values = torch.arange(6.0)
    assert torch.equal(fft_ifftshift(values), torch.fft.ifftshift(values))
