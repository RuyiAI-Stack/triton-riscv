import torch

from .detach_copy import _copy_values
from .randint import _philox_uniform


def cauchy_(self, median=0.0, sigma=1.0):
    u = _philox_uniform(self.numel(), self.device).view(self.shape)
    filled = median + sigma * torch.tan(torch.pi * (u - 0.5))
    self.copy_(_copy_values(filled.to(self.dtype)))
    return self


def cauchy(*size, median=0.0, sigma=1.0, dtype=torch.float32, device="cpu"):
    out = torch.empty(size, dtype=dtype, device=device)
    return cauchy_(out, median, sigma)
