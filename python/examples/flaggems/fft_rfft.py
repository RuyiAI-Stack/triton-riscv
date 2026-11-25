import torch


def _as_batch_last(tensor, dim):
    return torch.movedim(tensor, dim, -1).contiguous()


def fft_rfft(self, n=None, dim=-1, norm=None):
    from .fft import fft

    spec_in = _as_batch_last(self, dim)
    length = spec_in.shape[-1] if n is None else n
    if spec_in.shape[-1] < length:
        spec_in = torch.nn.functional.pad(spec_in, (0, length - spec_in.shape[-1]))
    elif spec_in.shape[-1] > length:
        spec_in = spec_in[..., :length]
    flat = spec_in.reshape(-1, length)
    spec = fft(flat)[..., : length // 2 + 1]
    if norm == "forward":
        spec = spec / length
    elif norm == "ortho":
        spec = spec / (length**0.5)
    return torch.movedim(spec.reshape(spec_in.shape[:-1] + (length // 2 + 1,)), -1, dim)
