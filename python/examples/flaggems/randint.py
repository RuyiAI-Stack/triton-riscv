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
    UNROLL = 4

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

    randint_kernel[grid](
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
