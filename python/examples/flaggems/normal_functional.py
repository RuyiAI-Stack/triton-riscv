import torch

from .randint import _philox_uniform


def _box_muller(n, device):
    u1 = _philox_uniform(n, device)
    u2 = _philox_uniform(n, device)
    return torch.sqrt(-2 * torch.log(u1)) * torch.cos(2 * torch.pi * u2)


def normal_functional(self, mean=0.0, std=1.0):
    return (mean + std * _box_muller(self.numel(), self.device).view(self.shape)).to(
        self.dtype
    )
