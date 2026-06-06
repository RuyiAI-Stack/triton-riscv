import torch
import triton
import triton.language as tl


@triton.jit
def uint_to_uniform_float(x):
    # conditions can be simplified
    # scale is ((2**23 - 1) / 2**23) * 2**(N_BITS - 1)
    if tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int32):
        # maximum value such that `MAX_INT * scale < 1.0` (with float rounding)
        x = x.to(tl.int32, bitcast=True)
        scale = 4.6566127342e-10
    else:
        tl.static_assert(
            tl.constexpr(x.dtype == tl.uint64) or tl.constexpr(x.dtype == tl.int64)
        )
        x = x.to(tl.int64, bitcast=True)
        scale = 1.0842020432385337e-19
    x = tl.where(x < 0, -x - 1, x)
    return x * scale


def philox_backend_seed_offset(increment, generator=None):
    increment = (int(increment) + 3) // 4 * 4
    if increment == 0:
        return 0, 0

    if generator is None:
        generator = torch.default_generator

    state_copy = generator.get_state()
    state_view = state_copy.view(torch.int64)

    if state_view.numel() != 2:
        values = torch.empty(increment, dtype=torch.int64).random_(generator=generator)
        return int(values[0].item()), 0

    seed = int(state_view[0].item())
    offset = int(state_view[1].item())
    state_view[1] += increment
    generator.set_state(state_copy)
    return seed, offset


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def rand_kernel(
    out_ptr,
    N,
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
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + off_0, r0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, r1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, r2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, r3, mask=off_3 < N, eviction_policy="evict_first")


UNROLL = 4


def rand(
    size,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    if dtype is None:
        dtype = torch.get_default_dtype()

    out = torch.empty(size, device=device, dtype=dtype)
    N = out.numel()
    if N == 0:
        return out

    BLOCK = 1024

    def grid_fn(meta):
        return (triton.cdiv(N, BLOCK * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)

    rand_kernel[grid_fn](out, N, philox_seed, philox_offset, BLOCK=BLOCK)
    return out
