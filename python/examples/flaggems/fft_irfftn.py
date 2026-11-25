import torch


def _as_batch_last(tensor, dim):
    return torch.movedim(tensor, dim, -1).contiguous()


def fft_irfftn(input, s=None, dim=None, norm=None):
    from .fft import fft

    if dim is None:
        dim = -1
    spec = _as_batch_last(input, dim)
    length = spec.shape[-1] * 2 - 2 if s is None else s[-1]
    full = torch.zeros(
        *spec.shape[:-1], length, dtype=torch.complex64, device=input.device
    )
    full[..., : spec.shape[-1]] = spec
    if length > spec.shape[-1]:
        mirror = torch.flip(spec[..., 1:-1].conj(), dims=(-1,))
        full[..., spec.shape[-1] :] = mirror
    flat = full.reshape(-1, length)
    time = fft(torch.conj(flat)).conj().real
    scale = length
    if norm == "forward":
        scale = 1.0
    elif norm == "ortho":
        scale = length**0.5
    time = time / scale
    return torch.movedim(time.reshape(full.shape), -1, dim)
