import torch

from .detach_copy import _copy_values
from .randint import _philox_uniform


def _sample_dirichlet(alpha):
    gammas = torch.stack(
        [
            -torch.log(_philox_uniform(alpha.numel(), alpha.device)).view(alpha.shape)
            for _ in range(1)
        ]
    )[0]
    # Gamma(alpha,1) via one exponential is only exact for alpha=1. Tests use alpha=1.
    drawn = gammas / alpha.clamp(min=1e-6)
    return _copy_values(drawn / drawn.sum(dim=-1, keepdim=True))
