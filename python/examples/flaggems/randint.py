import torch
import triton


def _philox_uniform(n, device):
    from .rand import philox_backend_seed_offset
    from .uniform import uniform_kernel

    out = torch.empty(n, dtype=torch.float32, device=device)
    if n == 0:
        return out
    block = 128
    increment = triton.cdiv(n, 4)
    seed, offset = philox_backend_seed_offset(increment)
    uniform_kernel[(triton.cdiv(n, block * 4),)](
        out, n, seed, offset, 0.0, 1.0, BLOCK=block
    )
    return out.clamp(1e-6, 1 - 1e-6)


def randint(
    low, high=None, size=None, *, dtype=torch.int64, device="cpu", generator=None
):
    _ = generator
    if high is None:
        high = low
        low = 0
    if isinstance(high, (tuple, list)):
        size = high
        high = low
        low = 0
    if size is None:
        raise TypeError("randint requires a size")
    n = 1
    for dim in size:
        n *= dim
    u = _philox_uniform(n, device)
    span = int(high) - int(low)
    values = (u * span).to(torch.int64) + int(low)
    return values.view(size).to(dtype)
