import torch

from .detach_copy import _copy_values
from .fft_irfftn import fft_irfftn


def istft(
    input,
    n_fft,
    hop_length=None,
    win_length=None,
    window=None,
    center=True,
    normalized=False,
    onesided=True,
    length=None,
    return_complex=False,
):
    _ = (normalized, onesided, return_complex)
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    frames = fft_irfftn(input, s=(n_fft,), dim=-2 if input.ndim > 1 else -1)
    # frames from a spectrogram are (..., freq, time) in torch.istft. Accept
    # (freq, time) and overlap-add along time.
    if frames.ndim == 1:
        return frames
    spec = input
    n_frames = spec.shape[-1]
    out_len = n_fft + hop_length * (n_frames - 1)
    signal = torch.zeros(out_len, dtype=frames.dtype, device=frames.device)
    window_sum = torch.zeros(out_len, dtype=frames.dtype, device=frames.device)
    win = torch.ones(n_fft, device=frames.device) if window is None else window
    time = fft_irfftn(spec.transpose(-1, -2), s=(n_fft,), dim=-1)
    for frame in range(n_frames):
        start = frame * hop_length
        signal[start : start + n_fft] += time[frame] * win
        window_sum[start : start + n_fft] += win.square()
    signal = signal / window_sum.clamp(min=1e-8)
    if center:
        signal = signal[n_fft // 2 : -n_fft // 2]
    if length is not None:
        signal = signal[:length]
    return _copy_values(signal)
