import triton
import triton.language as tl

from .rand import philox_backend_seed_offset


@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "p"])
def bernoulli_kernel(
    out_ptr,
    N,
    p,
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

    u0 = tl.uint_to_uniform_float(r0)
    u1 = tl.uint_to_uniform_float(r1)
    u2 = tl.uint_to_uniform_float(r2)
    u3 = tl.uint_to_uniform_float(r3)

    y0 = tl.where(u0 < p, 1.0, 0.0)
    y1 = tl.where(u1 < p, 1.0, 0.0)
    y2 = tl.where(u2 < p, 1.0, 0.0)
    y3 = tl.where(u3 < p, 1.0, 0.0)

    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    tl.store(out_ptr + off_0, y0, mask=off_0 < N)
    tl.store(out_ptr + off_1, y1, mask=off_1 < N)
    tl.store(out_ptr + off_2, y2, mask=off_2 < N)
    tl.store(out_ptr + off_3, y3, mask=off_3 < N)


UNROLL = 4


def bernoulli_(self, p=0.5, *, generator=None):
    N = self.numel()
    BLOCK = 512
    grid = (triton.cdiv(N, BLOCK * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    bernoulli_kernel[grid](self, N, p, philox_seed, philox_offset, BLOCK=BLOCK)
    return self
