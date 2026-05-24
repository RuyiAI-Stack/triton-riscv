import torch
import triton

from .randn import randn_kernel


UNROLL = 4


def _get_philox_seed_offset(increment, generator=None):
    if generator is not None:
        seed = generator.seed()
    else:
        seed = torch.initial_seed()
    offset = 0
    increment = (increment + 3) // 4 * 4
    return seed, offset


def normal_tensor_tensor(mean, std, *, generator=None):
    shape = torch.broadcast_tensors(
        torch.empty(mean.shape, device=mean.device),
        torch.empty(std.shape, device=std.device),
    )[0].shape
    device = mean.device
    out = torch.empty(shape, device=device, dtype=torch.float32)
    N = out.numel()
    if N == 0:
        return out

    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = _get_philox_seed_offset(
        increment, generator=generator
    )
    randn_kernel[grid](out, N, philox_seed, philox_offset, BLOCK=BLOCK)
    return out * std + mean


def normal_tensor_float(mean, std, *, generator=None):
    shape = mean.shape
    device = mean.device
    out = torch.empty(shape, device=device, dtype=torch.float32)
    N = out.numel()
    if N == 0:
        return out

    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = _get_philox_seed_offset(
        increment, generator=generator
    )
    randn_kernel[grid](out, N, philox_seed, philox_offset, BLOCK=BLOCK)
    return out * std + mean


def normal_float_tensor(mean, std, *, generator=None):
    shape = std.shape
    device = std.device
    out = torch.empty(shape, device=device, dtype=torch.float32)
    N = out.numel()
    if N == 0:
        return out

    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = _get_philox_seed_offset(
        increment, generator=generator
    )
    randn_kernel[grid](out, N, philox_seed, philox_offset, BLOCK=BLOCK)
    return out * std + mean


def normal(mean, std, *, generator=None):
    """Entry point matching torch.normal signature."""
    if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
        return normal_tensor_tensor(mean, std, generator=generator)
    elif isinstance(mean, torch.Tensor):
        return normal_tensor_float(mean, std, generator=generator)
    elif isinstance(std, torch.Tensor):
        return normal_float_tensor(mean, std, generator=generator)
    else:
        raise TypeError("normal: at least one of mean or std must be a Tensor")


def normal_(self, mean=0, std=1, *, generator=None):
    N = self.numel()
    if N == 0:
        return self

    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = _get_philox_seed_offset(
        increment, generator=generator
    )
    randn_kernel[grid](self, N, philox_seed, philox_offset, BLOCK=BLOCK)
    self.mul_(std).add_(mean)
    return self
