import torch
import triton
import triton.language as tl


@triton.jit
def _lgamma_pos(z):
    z = z.to(tl.float32) - 1.0
    x = 0.99999999999980993
    x += 676.5203681218851 / (z + 1.0)
    x += -1259.1392167224028 / (z + 2.0)
    x += 771.32342877765313 / (z + 3.0)
    x += -176.61502916214059 / (z + 4.0)
    x += 12.507343278686905 / (z + 5.0)
    x += -0.13857109526572012 / (z + 6.0)
    x += 9.9843695780195716e-6 / (z + 7.0)
    x += 1.5056327351493116e-7 / (z + 8.0)
    t = z + 7.5
    return 0.9189385332046727 + (z + 0.5) * tl.log(t) - t + tl.log(x)


@triton.jit
def gammainc_kernel(a_ptr, x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute in float32 for better precision
    a_f32 = a.to(tl.float32)
    x_f32 = x.to(tl.float32)

    # Handle edge cases
    # P(a, 0) = 0 for a > 0; NaN for a <= 0 or x < 0
    result = tl.where((a_f32 > 0.0) & (x_f32 >= 0.0), 0.0, float("nan"))

    # Regularized lower incomplete gamma function P(a, x) for x > 0
    # Using series expansion for small x and continued fraction for large x

    # Determine which method to use based on x and a
    # Series expansion is better when x < a + 1
    use_series = x_f32 < (a_f32 + 1.0)

    # Series expansion: P(a, x) = exp(-x) * x^a * sum_{n=0} x^n / Gamma(a+n+1)
    # with term recurrence: t_0 = 1/Gamma(a+1), t_n = t_{n-1} * x / (a+n)
    # Implemented via: sum_n starting from 1/a, then divide by Gamma(a) at the end.
    # sum = Gamma(a) * sum_{n=0} x^n / Gamma(a+n+1) = sum_{n=0} x^n / ((a)_n * a)
    # where (a)_n = a*(a+1)*...*(a+n-1) is the rising factorial.
    # So series_result = exp(-x) * x^a * sum / Gamma(a)
    series_sum = 0.0
    term = 1.0 / a_f32
    series_sum = term
    active = tl.full([BLOCK_SIZE], True, tl.int1)
    for i in range(1, 200):
        next_term = term * x_f32 / (a_f32 + tl.cast(i, tl.float32))
        series_sum += tl.where(active, next_term, 0.0)
        active = active & ~(tl.abs(next_term) < (tl.abs(series_sum) * 1e-10))
        term = tl.where(active, next_term, 0.0)

    # Divide by Gamma(a) to get the regularized value P(a, x)
    log_gamma_a = _lgamma_pos(a_f32)
    series_result = tl.exp(-x_f32 + a_f32 * tl.log(x_f32) - log_gamma_a) * series_sum

    # Lentz's continued fraction for Q(a,x) = Gamma(a,x)/Gamma(a)
    # for the large-x regime (x >= a + 1).
    #
    # CF = b_0 + a_1/(b_1 + a_2/(b_2 + ...))
    #   b_0 = x + 1 - a
    #   a_n = n(a - n),  b_n = x + 2n + 1 - a   (n >= 1)
    # Then Q = e^{-x} * x^a / (Gamma(a) * CF) and P = 1 - Q.
    tiny = 1e-30
    b0 = x_f32 + 1.0 - a_f32
    f_val = b0
    C_val = b0
    D_val = 0.0 * x_f32
    for i_val in range(1, 300):
        i_f = tl.cast(i_val, tl.float32)
        an = i_f * (a_f32 - i_f)
        bn = x_f32 + 2.0 * i_f + 1.0 - a_f32

        D_val = bn + an * D_val
        D_val = tl.where(tl.abs(D_val) < tiny, tiny, D_val)

        C_val = bn + an / C_val
        C_val = tl.where(tl.abs(C_val) < tiny, tiny, C_val)

        D_val = 1.0 / D_val
        delta = C_val * D_val
        f_val = f_val * delta

    log_gamma_a = _lgamma_pos(a_f32)
    log_q = a_f32 * tl.log(x_f32) - x_f32 - log_gamma_a - tl.log(f_val)
    q_val = tl.exp(log_q)
    q_val = tl.where(q_val > 1.0, 1.0, tl.where(q_val < 0.0, 0.0, q_val))
    frac_result = 1.0 - q_val

    # Combine results
    result = tl.where(
        (a_f32 > 0.0) & (x_f32 > 0.0),
        tl.where(use_series, series_result, frac_result),
        result,
    )
    inf = tl.full(a_f32.shape, float("inf"), dtype=tl.float32)
    finite_positive_a = (a_f32 > 0.0) & (a_f32 < inf)
    result = tl.where(finite_positive_a & (x_f32 == inf), 1.0, result)
    result = tl.where((a_f32 == inf) & (x_f32 >= 0.0) & (x_f32 < inf), 0.0, result)

    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


def _launch_gammainc(out: torch.Tensor, a: torch.Tensor, x: torch.Tensor):
    assert out.device == a.device == x.device, "All tensors must be on the same device"
    assert out.numel() == a.numel() == x.numel(), (
        "All tensors must have the same number of elements"
    )
    assert out.device == a.device == x.device, "All tensors must be on the same device"

    # Ensure floating point compute
    a_in = a
    x_in = x
    out_in = out

    if not a_in.is_floating_point():
        a_in = a_in.to(torch.get_default_dtype())
    if not x_in.is_floating_point():
        x_in = x_in.to(torch.get_default_dtype())

    # Cast input to match the desired output dtype if needed
    if a_in.dtype != out_in.dtype:
        a_in = a_in.to(out_in.dtype)
    if x_in.dtype != out_in.dtype:
        x_in = x_in.to(out_in.dtype)

    if out_in.numel() == 0:
        return out_in

    a_contig = a_in.contiguous()
    x_contig = x_in.contiguous()
    out_was_noncontig = not out_in.is_contiguous()
    out_contig = out_in.contiguous() if out_was_noncontig else out_in

    n_elements = out_contig.numel()
    # 1024 provides good occupancy for element-wise gammainc kernel
    BLOCK_SIZE = 1024

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    gammainc_kernel[grid](
        a_contig, x_contig, out_contig, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )

    if out_was_noncontig:
        out_in.copy_(out_contig)
    return out_in


def special_gammainc(a: torch.Tensor, x: torch.Tensor, *, out: torch.Tensor = None):
    if a.device != x.device:
        raise ValueError("gammainc: input tensors must be on the same device")

    a_broadcast, x_broadcast = torch.broadcast_tensors(a, x)

    if out is None:
        if not a_broadcast.is_floating_point():
            a_broadcast = a_broadcast.to(torch.get_default_dtype())
        if not x_broadcast.is_floating_point():
            x_broadcast = x_broadcast.to(torch.get_default_dtype())
        out_dtype = torch.promote_types(a_broadcast.dtype, x_broadcast.dtype)
        out = torch.empty(a_broadcast.shape, dtype=out_dtype, device=a_broadcast.device)
    else:
        if out.device != a_broadcast.device:
            raise ValueError("gammainc_out: output tensor must be on the input device")
        if not out.is_floating_point():
            raise TypeError("gammainc_out: output tensor must be a floating point type")
        if out.shape != a_broadcast.shape:
            out.resize_(a_broadcast.shape)
    _launch_gammainc(out, a_broadcast, x_broadcast)
    return out


def special_gammainc_out(a: torch.Tensor, x: torch.Tensor, out: torch.Tensor):
    return special_gammainc(a, x, out=out)
