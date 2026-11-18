import torch
import triton
import triton.language as tl

from .rand import philox_backend_seed_offset


@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
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

    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    c0 += i
    z = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, z, z)

    high_val = tl.full((), high, tl.uint64)
    r0_mod = (r0 % high_val).to(out_ptr.dtype.element_ty)
    r1_mod = (r1 % high_val).to(out_ptr.dtype.element_ty)
    r2_mod = (r2 % high_val).to(out_ptr.dtype.element_ty)
    r3_mod = (r3 % high_val).to(out_ptr.dtype.element_ty)

    start = pid.to(tl.uint64) * BLOCK * 4
    off0 = start + tl.arange(0, BLOCK)
    off1 = off0 + BLOCK
    off2 = off1 + BLOCK
    off3 = off2 + BLOCK

    tl.store(out_ptr + off0, r0_mod, mask=off0 < N)
    tl.store(out_ptr + off1, r1_mod, mask=off1 < N)
    tl.store(out_ptr + off2, r2_mod, mask=off2 < N)
    tl.store(out_ptr + off3, r3_mod, mask=off3 < N)


@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def wide_randint_kernel(
    out_ptr,
    N,
    high: tl.constexpr,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    counter = philox_offset.to(tl.uint64) + offsets.to(tl.uint64)
    c0 = counter.to(tl.uint32)
    c1 = (counter >> 32).to(tl.uint32)
    attempt = tl.full((BLOCK,), 0, tl.uint32)
    zero = tl.full((BLOCK,), 0, tl.uint32)
    r0, r1, _, _ = tl.philox(philox_seed.to(tl.int64), c0, c1, attempt, zero)
    random = (r0.to(tl.uint64) << 32) | r1.to(tl.uint64)
    high_val = tl.full((), high, tl.uint64)
    # Reject the incomplete residue cycle to avoid modulo bias, including for
    # wide int64 intervals. Retries use a separate Philox counter component.
    threshold = tl.full((), (1 << 64) % high, tl.uint64)
    rejected = (offsets < N) & (random < threshold)
    while tl.sum(rejected.to(tl.int32), axis=0) > 0:
        attempt += 1
        r0, r1, _, _ = tl.philox(philox_seed.to(tl.int64), c0, c1, attempt, zero)
        candidate = (r0.to(tl.uint64) << 32) | r1.to(tl.uint64)
        random = tl.where(rejected, candidate, random)
        rejected = (offsets < N) & (random < threshold)
    tl.store(out_ptr + offsets, random % high_val, mask=offsets < N)


def randint(
    high,
    size,
    *,
    generator=None,
    out=None,
    dtype=torch.int64,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=None,
):
    if high <= 0:
        raise RuntimeError(
            f"random_ expects 'from' to be less than 'to', but got from=0 >= to={high}"
        )

    if dtype is None:
        dtype = torch.int64

    if device is None:
        device = torch.device("cpu")

    if pin_memory is None:
        pin_memory = False

    if layout is None:
        layout = torch.strided

    N = 1
    for s in size:
        N *= s

    BLOCK_SIZE = 128  # matches philox 4-wide output for efficient random generation
    # Keep the source four-output path unless the range needs 64 random bits.
    UNROLL = 1 if high > 2**32 else 4
    kernel = wide_randint_kernel if high > 2**32 else randint_kernel

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)

    result = torch.empty(size, device=device, dtype=dtype, pin_memory=pin_memory)
    if N == 0:
        if out is not None:
            out.copy_(result)
            return out
        return result

    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )

    kernel[grid](
        result,
        N,
        high,
        philox_seed,
        philox_offset,
        BLOCK_SIZE,
    )

    if out is not None:
        out.copy_(result)
        return out
    return result
