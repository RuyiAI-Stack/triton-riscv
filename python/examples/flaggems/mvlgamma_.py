import torch
import triton
import triton.language as tl

# mvlgamma_ (in-place multivariate gamma): sum_{k=0}^{p-1} lgamma(x -
# k/2) plus p(p-1)/4 * log(pi) written back into x, the same
# construction as the mvlgamma port (12-term unroll under runtime p
# guards, gated Lanczos lgamma inlined -- libdevice is not available on
# this backend; reflection only for x < 0 because this backend's fp64
# sin carries ~1e-7 relative error; poles at 0 and the negative
# integers give +inf).  The compute dtype follows the input.  p is
# passed as an int32 tensor (loading it in the kernel avoids the int
# scalar-argument == 1 specialization hazard).  aten parity: mvlgamma_
# is elementwise and rejects p < 1 with "p has to be greater than or
# equal to 1".


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
def _lgamma(x):
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
    return y


@triton.jit
def mvlgamma_kernel_(
    x_ptr,
    n_elements,
    p_ptr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_c = x.to(COMPUTE_DTYPE)

    # Load p value
    p = tl.load(p_ptr)
    p_c = p.to(COMPUTE_DTYPE)

    # Constant term: p * (p - 1) / 4 * log(pi)
    result = p_c * (p_c - 1.0) * (1.1447298858494002 / 4.0)

    # Sum of lgamma terms: sum_{k=0}^{p-1} lgamma(x - k/2), unrolled
    # for p up to 12 and selected at runtime with tl.where
    t0 = _lgamma(x_c - 0.0)
    t1 = _lgamma(x_c - 0.5)
    t2 = _lgamma(x_c - 1.0)
    t3 = _lgamma(x_c - 1.5)
    t4 = _lgamma(x_c - 2.0)
    t5 = _lgamma(x_c - 2.5)
    t6 = _lgamma(x_c - 3.0)
    t7 = _lgamma(x_c - 3.5)
    t8 = _lgamma(x_c - 4.0)
    t9 = _lgamma(x_c - 4.5)
    t10 = _lgamma(x_c - 5.0)
    t11 = _lgamma(x_c - 5.5)

    sum_term = t0
    sum_term = tl.where(p >= 2, sum_term + t1, sum_term)
    sum_term = tl.where(p >= 3, sum_term + t2, sum_term)
    sum_term = tl.where(p >= 4, sum_term + t3, sum_term)
    sum_term = tl.where(p >= 5, sum_term + t4, sum_term)
    sum_term = tl.where(p >= 6, sum_term + t5, sum_term)
    sum_term = tl.where(p >= 7, sum_term + t6, sum_term)
    sum_term = tl.where(p >= 8, sum_term + t7, sum_term)
    sum_term = tl.where(p >= 9, sum_term + t8, sum_term)
    sum_term = tl.where(p >= 10, sum_term + t9, sum_term)
    sum_term = tl.where(p >= 11, sum_term + t10, sum_term)
    sum_term = tl.where(p >= 12, sum_term + t11, sum_term)

    result = result + sum_term
    tl.store(x_ptr + offsets, result.to(x_ptr.dtype.element_ty), mask=mask)


def _compute_dtype(x):
    return tl.float64 if x.dtype == torch.float64 else tl.float32


def mvlgamma_(*args, **kwargs):
    x = args[0]
    p = args[1] if len(args) > 1 else kwargs.get("p", 1)

    if not isinstance(x, torch.Tensor):
        raise TypeError("mvlgamma_ expects a torch.Tensor as the first argument")

    # aten parity: p < 1 raises aten's RuntimeError, not upstream's
    # ValueError
    if not isinstance(p, int) or p < 1:
        raise RuntimeError("p has to be greater than or equal to 1")

    # Limit p to 12 since the kernel unrolls the loop 12 times
    if p > 12:
        raise ValueError("p must be <= 12 for this implementation")

    # Create a 1D tensor to pass p to the kernel
    p_tensor = torch.tensor([p], dtype=torch.int32)

    # Handle non-contiguous tensors by operating on a contiguous
    # copy and copying back
    if not x.is_contiguous():
        y = x.contiguous()
        n_elements = y.numel()
        if n_elements == 0:
            return x
        grid = (triton.cdiv(n_elements, 1024),)
        mvlgamma_kernel_[grid](
            y,
            n_elements,
            p_tensor,
            COMPUTE_DTYPE=_compute_dtype(y),
            BLOCK_SIZE=1024,
        )
        x.copy_(y)
        return x

    n_elements = x.numel()
    if n_elements == 0:
        return x
    grid = (triton.cdiv(n_elements, 1024),)
    mvlgamma_kernel_[grid](
        x,
        n_elements,
        p_tensor,
        COMPUTE_DTYPE=_compute_dtype(x),
        BLOCK_SIZE=1024,
    )
    return x
