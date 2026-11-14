import torch
import triton
import triton.language as tl


@triton.jit
def _abs_u32(x):
    # abs of INT_MIN overflows tl.abs; wrapping in uint32 yields the correct
    # magnitude for every int32 input.
    ux = x.to(tl.uint32)
    return tl.where(x < 0, 0 - ux, ux)


@triton.jit
def _abs_u64(x):
    ux = x.to(tl.uint64)
    return tl.where(x < 0, 0 - ux, ux)


@triton.jit
def _ctz(x):
    # ffs is not lowered on this backend; ctz(x) equals log2 of the lowest
    # set bit, which log2 computes exactly for powers of two. Callers guard
    # x against zero.
    low = x & -x
    return tl.log2(low.to(tl.float32)).to(tl.int32)


@triton.jit
def _binary_gcd(ax, ay, normal, MAX_ITERS: tl.constexpr):
    zero_ax = ax == 0
    zero_ay = ay == 0
    res = tl.where(zero_ax, ay, ax)
    both_nonzero = normal & (~zero_ax) & (~zero_ay)
    common = _ctz(tl.where(both_nonzero, ax | ay, 1))
    u = tl.where(both_nonzero, ax >> _ctz(tl.where(both_nonzero, ax, 1)), ax)
    v = ay
    active = both_nonzero
    for _ in tl.static_range(MAX_ITERS):
        v_shifted = tl.where(active, v >> _ctz(tl.where(active, v, 1)), v)
        swap = active & (u > v_shifted)
        small = tl.where(swap, v_shifted, u)
        large = tl.where(swap, u, v_shifted)
        u = tl.where(active, small, u)
        v = tl.where(active, large - small, v)
        active = active & (v != 0)
    return tl.where(both_nonzero, u << common, res)


@triton.jit
def lcm_kernel_i32(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    ax_u = _abs_u32(x)
    ay_u = _abs_u32(y)
    gcd_val = _binary_gcd(ax_u, ay_u, mask, MAX_ITERS=62)
    # lcm = |x| / gcd * |y| (divide first to avoid overflow); 0 when gcd is 0.
    safe_gcd_u = tl.where(gcd_val == 0, 1, gcd_val)
    result_u = (ax_u // safe_gcd_u) * ay_u
    result = result_u.to(tl.int32)
    result = tl.where(gcd_val == 0, 0, result)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def lcm_kernel_i64(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    ax_u = _abs_u64(x)
    ay_u = _abs_u64(y)
    gcd_val = _binary_gcd(ax_u, ay_u, mask, MAX_ITERS=126)
    safe_gcd_u = tl.where(gcd_val == 0, 1, gcd_val)
    result_u = (ax_u // safe_gcd_u) * ay_u
    result = result_u.to(tl.int64)
    result = tl.where(gcd_val == 0, 0, result)
    tl.store(out_ptr + offsets, result, mask=mask)


def lcm(A: torch.Tensor, B: torch.Tensor):
    if A.dtype.is_floating_point or B.dtype.is_floating_point:
        # torch: '"lcm_cpu" not implemented for Float'.
        raise NotImplementedError('"lcm_cpu" not implemented for floating point inputs')
    if A.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        return torch.lcm(A, B)

    promoted = torch.promote_types(A.dtype, B.dtype)
    lhs = A if A.dtype == promoted else A.to(promoted)
    rhs = B if B.dtype == promoted else B.to(promoted)
    # Out-of-place: the result takes the broadcast shape (A may be smaller).
    lhs, rhs = torch.broadcast_tensors(lhs, rhs)
    lhs_c = lhs.contiguous()
    rhs_c = rhs.contiguous()
    out = torch.empty_like(lhs_c)
    numel = out.numel()
    if numel == 0:
        return out

    if promoted == torch.int64:
        grid = (triton.cdiv(numel, 256),)
        lcm_kernel_i64[grid](lhs_c, rhs_c, out, numel, BLOCK=256)
    else:
        grid = (triton.cdiv(numel, 512),)
        lcm_kernel_i32[grid](lhs_c, rhs_c, out, numel, BLOCK=512)

    return out
