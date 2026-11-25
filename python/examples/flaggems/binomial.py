import torch

from .randint import _philox_uniform


def binomial(count, prob, size=None, *, dtype=torch.float32, device="cpu"):
    if size is None:
        size = count.shape if torch.is_tensor(count) else prob.shape
    n = 1
    for dim in size:
        n *= dim
    trials = int(count) if not torch.is_tensor(count) else int(count.reshape(-1)[0])
    p = float(prob) if not torch.is_tensor(prob) else float(prob.reshape(-1)[0])
    draws = torch.stack([_philox_uniform(n, device) < p for _ in range(trials)])
    return draws.sum(dim=0).view(size).to(dtype)
