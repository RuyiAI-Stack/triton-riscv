import torch
import triton
import triton.language as tl

from .rand import philox_backend_seed_offset, uint_to_uniform_float

UNROLL = 4


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def randint_kernel(
    out_ptr,
    N,
    high,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)

    # Convert to uniform float in [0, 1)
    u0 = uint_to_uniform_float(r0)
    u1 = uint_to_uniform_float(r1)
    u2 = uint_to_uniform_float(r2)
    u3 = uint_to_uniform_float(r3)

    # Scale to [0, high) and convert to int32
    high_f = high * 1.0
    i0 = (u0 * high_f).to(tl.int32)
    i1 = (u1 * high_f).to(tl.int32)
    i2 = (u2 * high_f).to(tl.int32)
    i3 = (u3 * high_f).to(tl.int32)

    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    tl.store(out_ptr + off_0, i0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, i1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, i2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, i3, mask=off_3 < N, eviction_policy="evict_first")


def randint_like(
    self,
    high,
    *,
    dtype=None,
    generator=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    if device is None:
        device = self.device
    if dtype is None:
        dtype = self.dtype
    if isinstance(high, torch.Tensor):
        if high.ndim != 0:
            raise TypeError("high must be an integer or 0-d tensor")
        high = int(high.item())

    out = torch.empty_like(self, device=device, dtype=dtype)
    N = self.numel()
    if N == 0:
        return out

    def grid_fn(meta):
        return (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    BLOCK = 128  # matches philox 4-wide output for efficient random generation
    randint_kernel[grid_fn](out, N, high, philox_seed, philox_offset, BLOCK=BLOCK)
    return out
