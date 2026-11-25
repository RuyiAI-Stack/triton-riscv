import torch

from .detach_copy import _copy_values
from .randint import _philox_uniform


def geometric_(self, p):
    u = _philox_uniform(self.numel(), self.device).view(self.shape)
    filled = (
        torch.floor(
            torch.log(u)
            / torch.log(torch.tensor(1 - p, dtype=u.dtype, device=u.device))
        ).to(self.dtype)
        + 1
    )
    self.copy_(_copy_values(filled))
    return self


def geometric(*size, p=0.5, dtype=torch.float32, device="cpu"):
    out = torch.empty(size, dtype=dtype, device=device)
    return geometric_(out, p)
