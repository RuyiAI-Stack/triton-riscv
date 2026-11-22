import torch
import triton
import triton.language as tl

# igammac_ (in-place regularized upper incomplete gamma Q(a, x)):
# upstream's in-place entry wraps a 20-term power series for P with
# Q = 1 - P, which diverges for x >= a + 1 and cannot hold the suite's
# 1e-4 tolerance on that half of the domain.  This port keeps the
# entry's shape (kernel name igammac_func, result written back into
# self) but computes Q with the accurate two-branch construction from
# the igammac port: power series for P below a + 1 (Q = 1 - P) and
# Lentz's continued fraction for Q above, in the input's precision
# (fp64 inputs compute in double).  Boundary contract (probed on torch
# CPU): Q(a, 0) = 1, Q(a, inf) = 0, Q(inf, x) = 1, Q(inf, inf) = nan,
# x < 0 = nan (including -inf), a <= 0 = nan, nan propagates.


@triton.jit
def _lgamma_pos(x):
    z = x - 1.0
    a = 0.99999999999980993
    a = a + 676.5203681218851 / (z + 1.0)
    a = a + -1259.1392167224028 / (z + 2.0)
    a = a + 771.32342877765313 / (z + 3.0)
    a = a + -176.61502916214059 / (z + 4.0)
    a = a + 12.507343278686905 / (z + 5.0)
    a = a + -0.13857109526572012 / (z + 6.0)
    a = a + 9.9843695780195716e-06 / (z + 7.0)
    a = a + 1.5056327351493116e-07 / (z + 8.0)
    t = z + 7.5
    return 0.9189385332046727 + (z + 0.5) * tl.log(t) - t + tl.log(a)


@triton.jit
def igammac_func(
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

    a = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    a_f = a.to(COMPUTE_DTYPE)
    x_f = x.to(COMPUTE_DTYPE)

    # lgamma(a); garbage on the out-of-domain lanes is re-gated below
    log_gamma_a = _lgamma_pos(a_f)
    log_x_term = a_f * tl.log(x_f) - x_f - log_gamma_a

    # power series for P(a, x), Q = 1 - P (converges for x < a + 1)
    term = 1.0 / a_f
    series_sum = term
    for i in range(1, 50):
        term = term * x_f / (a_f + tl.cast(i, COMPUTE_DTYPE))
        series_sum = series_sum + term
    q_series = 1.0 - tl.exp(log_x_term) * series_sum

    # Lentz's continued fraction for Q(a, x) (converges for x >= a + 1)
    tiny = 1e-30
    b0 = x_f + 1.0 - a_f
    f_val = b0
    C_val = b0
    D_val = tl.zeros_like(x_f)
    for i in range(1, 50):
        i_f = tl.cast(i, COMPUTE_DTYPE)
        an = i_f * (a_f - i_f)
        bn = x_f + 2.0 * i_f + 1.0 - a_f

        D_val = bn + an * D_val
        D_val = tl.where(tl.abs(D_val) < tiny, tiny, D_val)
        C_val = bn + an / C_val
        C_val = tl.where(tl.abs(C_val) < tiny, tiny, C_val)

        D_val = 1.0 / D_val
        delta = C_val * D_val
        f_val = f_val * delta

    q_cf = tl.exp(log_x_term - tl.log(f_val))
    q_cf = tl.where(q_cf > 1.0, 1.0, tl.where(q_cf < 0.0, 0.0, q_cf))

    # per-element path selection
    use_series = x_f < (a_f + 1.0)
    result = tl.where(use_series, q_series, q_cf)

    # boundary handling
    result = tl.where(x_f != x_f, x_f, result)
    result = tl.where(a_f != a_f, a_f, result)
    result = tl.where(x_f < 0.0, float("nan"), result)
    result = tl.where(a_f <= 0.0, float("nan"), result)
    result = tl.where(x_f == float("inf"), 0.0, result)
    result = tl.where(a_f == float("inf"), 1.0, result)
    result = tl.where(
        (x_f == float("inf")) & (a_f == float("inf")), float("nan"), result
    )

    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def igammac_(A, B):
    # igammac_ (and thus this kernel) only supports floating point dtypes;
    # Half/BFloat16 are not supported. Reject them explicitly.
    supported_dtypes = (torch.float32, torch.float64)
    if A.dtype not in supported_dtypes or B.dtype not in supported_dtypes:
        raise RuntimeError(
            f"igammac_ Triton kernel supports dtypes {supported_dtypes}, "
            f"but got A.dtype={A.dtype}, B.dtype={B.dtype}"
        )

    # Handle empty tensors
    if A.numel() == 0:
        return A

    # Broadcast shapes
    output_shape = torch.broadcast_shapes(A.shape, B.shape)
    n_elements = output_shape.numel()
    if n_elements == 0:
        return A

    # Broadcast and make contiguous
    a_tensor = A.expand(output_shape).contiguous()
    x_tensor = B.expand(output_shape).contiguous()

    # Allocate output tensor
    result = torch.empty_like(a_tensor)

    compute_dtype = tl.float64 if a_tensor.dtype == torch.float64 else tl.float32
    grid = (triton.cdiv(n_elements, 1024),)
    igammac_func[grid](
        a_tensor,
        x_tensor,
        result,
        n_elements,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_SIZE=1024,
    )

    # Copy result back to A (in-place)
    A.copy_(result)
    return A
