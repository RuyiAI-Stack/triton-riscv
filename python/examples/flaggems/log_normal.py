import torch

from .randint import _philox_uniform


def _box_muller(n, device):
    u1 = _philox_uniform(n, device)
    u2 = _philox_uniform(n, device)
    return torch.sqrt(-2 * torch.log(u1)) * torch.cos(2 * torch.pi * u2)


def log_normal(*size, mean=0.0, std=1.0, dtype=torch.float32, device="cpu"):
    n = 1
    for dim in size:
        n *= dim
    normal = _box_muller(n, device)
    return torch.exp(mean + std * normal).view(size).to(dtype)
