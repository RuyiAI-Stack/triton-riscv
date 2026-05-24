import functools
import operator

import torch
import triton
import triton.language as tl


@triton.jit
def uint_to_uniform_float(x):
    if tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int32):
        # maximum value such that `MAX_INT * scale < 1.0` (with float rounding)
        x = x.to(tl.int32, bitcast=True)
        scale = 4.6566127342e-10
    else:
        tl.static_assert(
            tl.constexpr(x.dtype == tl.uint64)
            or tl.constexpr(x.dtype == tl.int64)
        )
        x = x.to(tl.int64, bitcast=True)
        scale = 1.0842020432385337e-19
    x = tl.where(x < 0, -x - 1, x)
    return x * scale


def volume(shape: tuple[int]) -> int:
    return functools.reduce(operator.mul, shape, 1)


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def uniform_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    from_,
    to,
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
    r0 = uint_to_uniform_float(r0) * (to - from_) + from_
    r1 = uint_to_uniform_float(r1) * (to - from_) + from_
    r2 = uint_to_uniform_float(r2) * (to - from_) + from_
    r3 = uint_to_uniform_float(r3) * (to - from_) + from_
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(
        out_ptr + off_0, r0, mask=off_0 < N, eviction_policy="evict_first"
    )
    tl.store(
        out_ptr + off_1, r1, mask=off_1 < N, eviction_policy="evict_first"
    )
    tl.store(
        out_ptr + off_2, r2, mask=off_2 < N, eviction_policy="evict_first"
    )
    tl.store(
        out_ptr + off_3, r3, mask=off_3 < N, eviction_policy="evict_first"
    )


UNROLL = 4


def uniform_(self, from_=0.0, to=1.0, *, generator=None):
    N = volume(self.shape)

    BLOCK = 128

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed = torch.randint(
        0, 2**32 - 1, (1,), dtype=torch.int64, device="cpu"
    ).item()
    philox_offset = increment

    uniform_kernel[grid](
        self, N, philox_seed, philox_offset, from_, to, BLOCK=BLOCK
    )
    return self


def uniform(size, from_=0.0, to=1.0, dtype=torch.float32, device="cpu"):
    out = torch.empty(size, dtype=dtype, device=device)
    return uniform_(out, from_=from_, to=to)
