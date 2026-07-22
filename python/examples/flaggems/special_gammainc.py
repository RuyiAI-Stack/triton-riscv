import torch
import triton
import triton.language as tl


@triton.jit
def _lgamma_pos(z):
    z = z.to(tl.float64) - 1.0
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
def _gammainc_continued_fraction(a, x, active):
    # Lentz's fraction for Q(a, x), on x >= a + 1. Inactive lanes
    # use benign inputs and do not participate in convergence decisions.
    a = tl.where(active, a, 0.5)
    x = tl.where(active, x, 2.0)
    fraction = x + 1.0 - a
    c = fraction
    d = tl.full(x.shape, 0.0, tl.float64)
    i = 1
    # Bound convergence so unsupported extreme inputs cannot loop indefinitely.
    while (i < 65536) & (tl.sum(active.to(tl.int32), 0) > 0):
        an = i * (a - i)
        bn = x + 2.0 * i + 1.0 - a
        d = bn + an * d
        d = tl.where(tl.abs(d) < 1e-300, 1e-300, d)
        c = bn + an / c
        c = tl.where(tl.abs(c) < 1e-300, 1e-300, c)
        d = 1.0 / d
        delta = c * d
        fraction = tl.where(active, fraction * delta, fraction)
        active = active & (tl.abs(delta - 1.0) > 1e-14)
        i += 1
    # Do not present a truncated, unconverged fraction as a valid result.
    return tl.where(active, float("nan"), fraction)


@triton.jit
def gammainc_kernel(a_ptr, x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Double intermediates reduce cancellation in a*log(x) - x - lgamma(a).
    a_compute = a.to(tl.float64)
    x_compute = x.to(tl.float64)

    # Handle edge cases
    # P(a, 0) = 0 for a > 0, P(0, x) = 1 for x > 0; P(0, 0) is NaN.
    result = tl.where((a_compute > 0.0) & (x_compute >= 0.0), 0.0, float("nan"))
    result = tl.where((a_compute == 0.0) & (x_compute > 0.0), 1.0, result)

    # Regularized lower incomplete gamma function P(a, x) for x > 0
    # Using series expansion for small x and continued fraction for large x

    # Determine which method to use based on x and a
    # Series expansion is better when x < a + 1
    use_series = x_compute < (a_compute + 1.0)
    finite_positive = mask & (a_compute > 0.0) & (a_compute < float("inf"))
    finite_positive = finite_positive & (x_compute > 0.0) & (x_compute < float("inf"))
    iterative = finite_positive

    # Normalize the series by Gamma(a+1), so its first term is 1.
    # This avoids both 1/a overflow and cancellation in Lanczos's a-1
    # argument when a is close to zero.
    active = iterative & use_series
    term = tl.where(active, 1.0, 0.0).to(tl.float64)
    series_sum = term
    i = 1
    # In this branch x < a + 1, so successive terms decrease. Stop each
    # lane only when its geometric upper bound on the remaining tail is small.
    # Bound convergence so unsupported extreme inputs cannot loop indefinitely.
    while (i < 65536) & (tl.sum(active.to(tl.int32), 0) > 0):
        ratio = x_compute / (a_compute + i)
        next_term = term * ratio
        series_sum += tl.where(active, next_term, 0.0)
        tail_bound = tl.abs(next_term) * ratio / (1.0 - ratio)
        active = active & (tail_bound > tl.abs(series_sum) * 1e-14)
        term = tl.where(active, next_term, 0.0)
        i += 1

    log_gamma_a_plus_one = _lgamma_pos(a_compute + 1.0)
    series_result = (
        tl.exp(-x_compute + a_compute * tl.log(x_compute) - log_gamma_a_plus_one)
        * series_sum
    )
    series_result = tl.where(active, float("nan"), series_result)

    f_val = _gammainc_continued_fraction(a_compute, x_compute, iterative & ~use_series)
    log_gamma_a = log_gamma_a_plus_one - tl.log(a_compute)
    log_q = a_compute * tl.log(x_compute) - x_compute - log_gamma_a - tl.log(f_val)
    q_val = tl.exp(log_q)
    q_val = tl.where(q_val > 1.0, 1.0, tl.where(q_val < 0.0, 0.0, q_val))
    frac_result = 1.0 - q_val

    # Combine results
    result = tl.where(
        (a_compute > 0.0) & (x_compute > 0.0),
        tl.where(use_series, series_result, frac_result),
        result,
    )
    inf = tl.full(a_compute.shape, float("inf"), dtype=tl.float64)
    finite_positive_a = (a_compute > 0.0) & (a_compute < inf)
    result = tl.where(finite_positive_a & (x_compute == inf), 1.0, result)
    result = tl.where(
        (a_compute == inf) & (x_compute >= 0.0) & (x_compute < inf), 0.0, result
    )

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
