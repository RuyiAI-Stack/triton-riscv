import torch
import triton
import triton.language as tl


@triton.jit
def uint_to_uniform_float(x):
    """Numerically stable function to convert a random uint into a random float uniformly sampled in [0, 1)."""
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


@triton.jit
def dropout_forward_kernel(
    X,
    Y,
    dropout_mask,
    N,
    p,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    UNROLL: tl.constexpr = 4  # philox generate 128 random bits at a time
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

    mask0 = r0 > p
    mask1 = r1 > p
    mask2 = r2 > p
    mask3 = r3 > p

    # We multiply by 1 / (1 - p) to preserve expected values
    scale = 1.0 / (1.0 - p)

    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    x0 = tl.load(X + off_0, mask=off_0 < N, other=0.0, eviction_policy="evict_first")
    x1 = tl.load(X + off_1, mask=off_1 < N, other=0.0, eviction_policy="evict_first")
    x2 = tl.load(X + off_2, mask=off_2 < N, other=0.0, eviction_policy="evict_first")
    x3 = tl.load(X + off_3, mask=off_3 < N, other=0.0, eviction_policy="evict_first")

    y0 = x0 * scale * mask0  # tl.where(mask0, x0 * p, 0.0)
    y1 = x1 * scale * mask1  # tl.where(mask1, x1 * p, 0.0)
    y2 = x2 * scale * mask2  # tl.where(mask2, x2 * p, 0.0)
    y3 = x3 * scale * mask3  # tl.where(mask3, x3 * p, 0.0)

    tl.store(
        dropout_mask + off_0,
        mask0.to(tl.int8),
        mask=off_0 < N,
        eviction_policy="evict_first",
    )
    tl.store(
        dropout_mask + off_1,
        mask1.to(tl.int8),
        mask=off_1 < N,
        eviction_policy="evict_first",
    )
    tl.store(
        dropout_mask + off_2,
        mask2.to(tl.int8),
        mask=off_2 < N,
        eviction_policy="evict_first",
    )
    tl.store(
        dropout_mask + off_3,
        mask3.to(tl.int8),
        mask=off_3 < N,
        eviction_policy="evict_first",
    )

    tl.store(Y + off_0, y0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(Y + off_1, y1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(Y + off_2, y2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(Y + off_3, y3, mask=off_3 < N, eviction_policy="evict_first")


@triton.jit
def dropout_backward_kernel(
    DY,
    DX,
    dropout_mask,
    N,
    scale,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N
    m = tl.load(
        dropout_mask + offset,
        mask=mask,
        other=0,
        eviction_policy="evict_first",
    )
    dy = tl.load(DY + offset, mask=mask, other=0, eviction_policy="evict_first")
    dx = dy * (m != 0).to(tl.float32) * scale
    tl.store(DX + offset, dx, mask=mask, eviction_policy="evict_first")


UNROLL = 4


def dropout(input, p, train=True):
    if not train or p == 0:
        out = input.clone()
        mask = torch.ones_like(input, dtype=torch.bool)
        return out, mask
    if p == 1:
        out = torch.zeros_like(input)
        mask = torch.zeros_like(input, dtype=torch.bool)
        return out, mask
    assert p > 0.0 and p < 1.0, "p must be in (0, 1)"

    input = input.contiguous()
    out = torch.empty_like(input)
    mask = torch.empty_like(input, dtype=torch.uint8)
    N = input.numel()

    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK * UNROLL),)

    # We need a seed and offset. We can use a random seed for now.
    philox_seed = torch.initial_seed()
    philox_offset = 0
    dropout_forward_kernel[grid](
        input, out, mask, N, p, philox_seed, philox_offset, BLOCK=BLOCK
    )
    return out, mask.to(torch.bool)


def dropout_backward(grad_output, mask, scale):
    grad_output = grad_output.contiguous()
    mask = mask.contiguous().to(torch.uint8)
    grad_input = torch.empty_like(grad_output)
    N = grad_output.numel()

    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)

    dropout_backward_kernel[grid](grad_output, grad_input, mask, N, scale, BLOCK=BLOCK)
    return grad_input
