import torch
import triton
import triton.language as tl

# lgamma / lgamma_ via the Lanczos approximation (g = 7, n = 9) with
# the reflection formula below 0 -- the repo's special_gammaln port
# reflects below 0.5, but this backend's fp64 sin/tl.cos carry ~1e-7
# relative error (measured), which the fp64 suite's 1e-9 tolerance
# cannot absorb; the Lanczos sum itself stays at double ulp on the
# whole positive axis (measured 7.6e-15 worst relative below 0.5), so
# the sin-dependent branch runs only where it is mathematically
# required, x < 0 (an fp32-only domain in this suite).  The compute
# dtype follows the input: fp64 inputs evaluate in double precision
# (upstream dispatches to libdevice lgamma, which is not available on
# this backend).  Boundary contract (probed on torch CPU): poles at 0
# and the negative integers give +inf (0.0 and -0.0 both), both
# infinities give +inf, nan stays nan, lgamma(1) = lgamma(2) = 0
# exactly (aten returns exact zeros there, so the anchors are gated
# rather than left to the Lanczos residual).


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
def lgamma_func(
    x_ptr,
    out_ptr,
    n_elements,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(COMPUTE_DTYPE)

    pi = 3.141592653589793
    y = tl.where(
        x < 0.0,
        tl.log(pi / tl.abs(tl.sin(pi * x))) - _lgamma_pos(1.0 - x),
        _lgamma_pos(x),
    )
    # exact anchors: aten's lgamma returns exact zeros at 1 and 2
    y = tl.where(x == 1.0, 0.0, y)
    y = tl.where(x == 2.0, 0.0, y)
    y = tl.where(x == 0.0, float("inf"), y)
    y = tl.where(x < 0.0, tl.where(x == tl.floor(x), float("inf"), y), y)
    y = tl.where(x == float("inf"), float("inf"), y)
    y = tl.where(x == float("-inf"), float("inf"), y)
    y = tl.where(x != x, x, y)

    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


def _compute_dtype(x):
    return tl.float64 if x.dtype == torch.float64 else tl.float32


def lgamma(A):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    if n_elements > 0:
        grid = (triton.cdiv(n_elements, 1024),)
        lgamma_func[grid](
            A_c,
            out,
            n_elements,
            COMPUTE_DTYPE=_compute_dtype(A_c),
            BLOCK_SIZE=1024,
        )
    return out.view_as(A)


def lgamma_(A):
    # Handle non-contiguous tensors by operating on a contiguous
    # copy and copying back
    if not A.is_contiguous():
        y = A.contiguous()
        n_elements = y.numel()
        if n_elements > 0:
            grid = (triton.cdiv(n_elements, 1024),)
            lgamma_func[grid](
                y,
                y,
                n_elements,
                COMPUTE_DTYPE=_compute_dtype(y),
                BLOCK_SIZE=1024,
            )
        A.copy_(y)
        return A

    n_elements = A.numel()
    if n_elements > 0:
        grid = (triton.cdiv(n_elements, 1024),)
        lgamma_func[grid](
            A,
            A,
            n_elements,
            COMPUTE_DTYPE=_compute_dtype(A),
            BLOCK_SIZE=1024,
        )
    return A
