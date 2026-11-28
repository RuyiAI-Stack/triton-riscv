"""``_standard_gamma_grad`` port (gradient of the Gamma sampler).

Port notes (triton-riscv, vs the upstream flag_gems file):
- ``tl_extra_shim.pow`` (libdevice pow) does not lower on this backend;
  every use has a strictly positive base (``x``, and the convex term
  ``x - alpha - alpha*log(x/alpha) >= 0``), so ``pow(b, e)`` is replaced
  by ``exp(e * log(b))`` — same value up to fp32 rounding, within the
  suite's 1e-3 band.
- pointwise_dynamic is replaced by a flat kernel with an IS_FP64
  constexpr for the double-precision path; the module-level
  ``_standard_gamma_grad_func`` name is kept (it is part of the tested
  surface).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _pow_pos(base, expo):
    return tl.exp(expo * tl.log(base))


@triton.jit
def _digamma_one(alpha):
    # digamma(x) for x > 0, mirroring ATen's ``digamma_one``: push x up to
    # >= 10 by the recurrence psi(x) = psi(x+1) - 1/x, then the asymptotic
    # expansion. Runs in the caller's dtype (fp32 or fp64).
    x = alpha
    result = tl.zeros_like(x)
    # alpha is positive; 20 recurrence steps cover alpha down to ~1e-2.
    for _ in range(20):
        need = x < 10.0
        result = result - tl.where(need, 1.0 / x, 0.0)
        x = tl.where(need, x + 1.0, x)

    z = 1.0 / (x * x)
    poly = 8.33333333333333333333e-2
    poly = poly * z + (-2.10927960927960927961e-2)
    poly = poly * z + 7.57575757575757575758e-3
    poly = poly * z + (-4.16666666666666666667e-3)
    poly = poly * z + 3.96825396825396825397e-3
    poly = poly * z + (-8.33333333333333333333e-3)
    poly = poly * z + 8.33333333333333333333e-2
    y = z * poly
    return result + tl.log(x) - 0.5 / x - y


@triton.jit
def _standard_gamma_grad_one(alpha, x):
    # Reparameterized gradient -(d/dalpha cdf(x; alpha)) / pdf(x; alpha),
    # ported from ATen's ``standard_gamma_grad_one`` (Distributions.h).
    # ``alpha`` and ``x`` carry the compute dtype (fp32 or fp64).

    # --- Branch 1: Taylor series expansion for small x (x < 0.8). ---
    numer = tl.full(alpha.shape, 1.0, alpha.dtype)
    denom = alpha
    series1 = numer / denom
    series2 = numer / (denom * denom)
    for i in range(1, 6):
        numer = numer * (-x / i)
        denom = denom + 1.0
        series1 = series1 + numer / denom
        series2 = series2 + numer / (denom * denom)
    pow_x_alpha = _pow_pos(x, alpha)
    gamma_pdf = _pow_pos(x, alpha - 1.0) * tl.exp(-x)
    gamma_cdf = pow_x_alpha * series1
    gamma_cdf_alpha = (
        tl.log(x) - _digamma_one(alpha)
    ) * gamma_cdf - pow_x_alpha * series2
    small = -gamma_cdf_alpha / gamma_pdf
    small = tl.where(small != small, 0.0, small)

    # --- Branch 2: Rice saddle point expansion for large alpha (alpha > 8). ---
    # Sub-case 2a: near the mode (0.9 alpha <= x <= 1.1 alpha).
    numer_1 = 1.0 + 24.0 * alpha * (1.0 + 12.0 * alpha)
    numer_2 = (
        1440.0 * (alpha * alpha)
        + 6.0 * x * (53.0 - 120.0 * x)
        - 65.0 * x * x / alpha
        + alpha * (107.0 + 3600.0 * x)
    )
    denom_a = 1244160.0 * (alpha * alpha) * (alpha * alpha)
    near_mode = numer_1 * numer_2 / denom_a

    # Sub-case 2b: away from the mode.
    denom_b = tl.sqrt(8.0 * alpha)
    term2 = denom_b / (alpha - x)
    neg_three_halves = tl.full(alpha.shape, -1.5, alpha.dtype)
    term3 = _pow_pos(x - alpha - alpha * tl.log(x / alpha), neg_three_halves)
    term23 = tl.where(x < alpha, term2 - term3, term2 + term3)
    term1 = tl.log(x / alpha) * term23 - tl.sqrt(2.0 / alpha) * (alpha + x) / (
        (alpha - x) * (alpha - x)
    )
    stirling = 1.0 + 1.0 / (12.0 * alpha) * (1.0 + 1.0 / (24.0 * alpha))
    away = -stirling * (x * term1) / denom_b

    is_near = (0.9 * alpha <= x) & (x <= 1.1 * alpha)
    large = tl.where(is_near, near_mode, away)

    # --- Branch 3: bivariate rational approximation (default). ---
    u = tl.log(x / alpha)
    v = tl.log(alpha)
    cv0 = 0.16009398 + u * (0.53487893 + u * 0.040121004)
    cv1 = -0.094634809 + u * (0.1298071 + u * (-0.0065914022))
    cv2 = 0.025146376 + u * (0.065735949 + u * (-0.0026286047))
    cv3 = -0.0030648343 + u * (-0.0015649758 + u * (-0.0013441777))
    cv4 = 1.0 + u * (0.16639465 + u * 0.017050642)
    cv5 = 0.32668115 + u * (0.020070113 + u * (-0.0021309326))
    cv6 = 0.10406089 + u * (-0.0035938915 + u * 0.00085092367)
    cv7 = 0.0014179084 + u * (-0.00058392623 + u * (-1.5247877e-07))
    p = cv0 + v * (cv1 + v * (cv2 + v * cv3))
    q = cv4 + v * (cv5 + v * (cv6 + v * cv7))
    rational = tl.exp(p / q)

    result = tl.where(x < 0.8, small, tl.where(alpha > 8.0, large, rational))
    return result


@triton.jit
def _standard_gamma_grad_kernel(
    a_ptr, x_ptr, out_ptr, n_elements, IS_FP64: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
    if IS_FP64:
        av = a.to(tl.float64)
        xv = x.to(tl.float64)
    else:
        av = a.to(tl.float32)
        xv = x.to(tl.float32)
    out = _standard_gamma_grad_one(av, xv)
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


def _standard_gamma_grad_func(self_grad, output):
    # Preserve double precision for fp64 inputs; otherwise compute in fp32
    # (mirroring ATen, which computes in the input's floating dtype).
    result = torch.empty_like(self_grad)
    n = self_grad.numel()
    if n == 0:
        return result
    a = self_grad.contiguous()
    x = output.contiguous()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _standard_gamma_grad_kernel[grid](
        a,
        x,
        result,
        n,
        IS_FP64=self_grad.dtype == torch.float64,
        BLOCK=BLOCK,
    )
    return result


def standard_gamma_grad(self_grad, output):
    logger.debug("GEMS STANDARD_GAMMA_GRAD")
    # Match the native _standard_gamma_grad dtype contract: both inputs must
    # be floating point and share the same dtype.
    if not torch.is_floating_point(self_grad) or not torch.is_floating_point(output):
        raise RuntimeError(
            "_standard_gamma_grad: expected floating point tensors, got "
            f"self_grad={self_grad.dtype}, output={output.dtype}"
        )
    if self_grad.dtype != output.dtype:
        raise RuntimeError(
            "_standard_gamma_grad: expected self_grad and output to have the "
            f"same dtype, got {self_grad.dtype} and {output.dtype}"
        )
    return _standard_gamma_grad_func(self_grad, output)
