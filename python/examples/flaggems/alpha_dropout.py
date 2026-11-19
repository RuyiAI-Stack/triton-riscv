import math

import torch
import triton
import triton.language as tl

# alpha_dropout (SELU-style): during training each element is dropped with
# probability p; kept elements take the variance-preserving affine
# a * x + b and dropped elements collapse to the constant a * alpha' + b
# (alpha' = -1.7580993408473766). Probed on torch CPU: eval and p == 0 are
# pass-throughs returning the input object, and p == 1 returns x * 0 with
# the signed zero preserved.
#
# The keep mask is drawn from the torch RNG (deterministic under
# torch.manual_seed) and the affine application runs in the Triton kernel.

_ALPHA = 1.6732632423543772848170429916717
_SCALE = 1.0507009873554804934193349852946
_ALPHA_P = -_ALPHA * _SCALE


def _alpha_dropout_affine(p):
    if p == 0.0:
        return 1.0, 0.0
    keep = 1.0 - p
    a = 1.0 / math.sqrt(keep * (1.0 + p * _ALPHA_P * _ALPHA_P))
    b = -a * p * _ALPHA_P
    return a, b


@triton.jit
def _alpha_dropout_kernel(
    x_ptr, keep_ptr, out_ptr, n_elements, a, b, dropped, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    keep = tl.load(keep_ptr + offsets, mask=mask, other=0)
    res = tl.where(keep != 0, a * x + b, dropped)
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _apply(x, keep, a, b, dropped):
    out = torch.empty_like(x)
    n = x.numel()
    if n:
        _alpha_dropout_kernel[(triton.cdiv(n, 1024),)](
            x, keep, out, n, float(a), float(b), float(dropped), BLOCK=1024
        )
    return out


def alpha_dropout(input, p=0.5, train=True):
    if not train or p == 0.0:
        return input
    if p == 1.0:
        return input * 0.0
    a, b = _alpha_dropout_affine(p)
    dropped = a * _ALPHA_P + b
    x = input.contiguous()
    keep = (torch.rand(x.shape, dtype=torch.float32, device=x.device) >= p).to(
        torch.uint8
    )
    return _apply(x, keep, a, b, dropped)
