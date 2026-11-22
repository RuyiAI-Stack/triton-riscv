import torch
import triton
import triton.language as tl

# igamma (regularized lower incomplete gamma P(a, x)), the upstream
# construction: the power series below a + 1 and Lentz's continued
# fraction for Q(a, x) above, seeded with the Lanczos (g = 7, n = 9)
# lgamma with reflection below 0.5.  Both convergence loops run a fixed
# 60 iterations and are selected per element with tl.where (the upstream
# block-uniform skip over the continued fraction does not lower on this
# backend).  The compute dtype follows the input: fp64 inputs run the
# whole construction in double precision.  Boundary contract (probed on
# torch CPU): P(a, 0) = 0, P(a, inf) = 1, x < 0 = nan, a <= 0 = nan,
# nan propagates.  log(pi) is spelled on the runtime argument: log() of
# a foldable constant is mis-folded to log2 on this backend.


@triton.jit
def igamma_kernel(
    a_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    a_f = a.to(COMPUTE_DTYPE)
    x_f = x.to(COMPUTE_DTYPE)

    pi = 3.1415926535897932384626433832795028841971

    # lgamma(a) via Lanczos, reflected below 0.5
    reflect_mask = a_f < 0.5
    a_refl = tl.where(reflect_mask, 1.0 - a_f, a_f)
    z = a_refl - 1.0
    p = (
        0.99999999999980993
        + 676.5203681218851 / (z + 1.0)
        + -1259.1392167224028 / (z + 2.0)
        + 771.32342877765313 / (z + 3.0)
        + -176.61502916214059 / (z + 4.0)
        + 12.507343278686905 / (z + 5.0)
        + -0.13857109526572012 / (z + 6.0)
        + 9.9843695780195716e-6 / (z + 7.0)
        + 1.5056327351493116e-7 / (z + 8.0)
    )
    t = z + 7.5
    log_gamma_a = 0.9189385332046727 + (z + 0.5) * tl.log(t) - t + tl.log(p)
    log_gamma_a = tl.where(
        reflect_mask,
        tl.log(pi / tl.abs(tl.sin(pi * a_f))) - log_gamma_a,
        log_gamma_a,
    )

    # common prefactor ax = x^a * e^{-x} / Gamma(a)
    ax = tl.exp(a_f * tl.log(x_f) - x_f - log_gamma_a)

    # power series (x < a + 1): P = ax * sum x^n / (a...(a+n))
    small_x = x_f < (a_f + 1.0)
    series_term = 1.0 / a_f
    series_result = series_term
    ap = a_f
    for _ in range(60):
        ap = ap + 1.0
        series_term = series_term * x_f / ap
        series_result = series_result + series_term
    series_res = ax * series_result

    # Lentz continued fraction for Q (x >= a + 1); P = 1 - Q
    fpmin = 1e-30
    b = x_f + 1.0 - a_f
    c = b * 0.0 + 1.0e30
    d = 1.0 / b
    h = d
    n = 1.0
    for _ in range(60):
        an = -n * (n - a_f)
        b = b + 2.0
        d = an * d + b
        d = tl.where(tl.abs(d) < fpmin, fpmin, d)
        c = b + an / c
        c = tl.where(tl.abs(c) < fpmin, fpmin, c)
        d = 1.0 / d
        delta = c * d
        h = h * delta
        n = n + 1.0
    cf_res = 1.0 - ax * h

    result = tl.where(small_x, series_res, cf_res)
    result = tl.where(x_f == 0.0, 0.0, result)
    result = tl.minimum(result, 1.0)
    result = tl.maximum(result, 0.0)
    result = tl.where(x_f == float("inf"), 1.0, result)
    result = tl.where(x_f < 0.0, float("nan"), result)
    result = tl.where(a_f <= 0.0, float("nan"), result)
    result = tl.where(x_f != x_f, x_f, result)
    result = tl.where(a_f != a_f, a_f, result)

    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def igamma(a, x):
    if not isinstance(a, torch.Tensor):
        raise TypeError("igamma expects a torch.Tensor as the first argument")
    if not isinstance(x, torch.Tensor):
        raise TypeError("igamma expects a torch.Tensor as the second argument")

    if a.shape != x.shape:
        raise ValueError("a and x must have the same shape")

    if not a.is_contiguous():
        a = a.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()

    n_elements = a.numel()
    if n_elements == 0:
        return torch.empty_like(a)

    output = torch.empty_like(a)
    compute_dtype = tl.float64 if a.dtype == torch.float64 else tl.float32
    block_size = 512
    grid = (triton.cdiv(n_elements, block_size),)
    igamma_kernel[grid](
        a,
        x,
        output,
        n_elements,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_SIZE=block_size,
    )
    return output
