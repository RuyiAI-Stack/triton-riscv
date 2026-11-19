import math

import torch
import triton
import triton.language as tl

# In-place alpha_dropout: input is mutated and returned.
# eval / p == 0 are no-ops on the input object; p == 1 writes signed zeros
# via mul_(0) (probed on torch CPU: the sign bit survives).

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
def _alpha_dropout_inplace_kernel(
    x_ptr, keep_ptr, out_ptr, n_elements, a, b, dropped, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    keep = tl.load(keep_ptr + offsets, mask=mask, other=0)
    res = tl.where(keep != 0, a * x + b, dropped)
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def alpha_dropout_(input, p=0.5, train=True):
    """In-place alpha_dropout; eval and p == 0 return the input unchanged."""
    if not train or p == 0.0:
        return input
    if p == 1.0:
        return input.mul_(0.0)
    a, b = _alpha_dropout_affine(p)
    dropped = a * _ALPHA_P + b
    assert input.is_contiguous(), "alpha_dropout_ requires a contiguous tensor"
    keep = (torch.rand(input.shape, dtype=torch.float32, device=input.device) >= p).to(
        torch.uint8
    )
    n = input.numel()
    if n:
        _alpha_dropout_inplace_kernel[(triton.cdiv(n, 1024),)](
            input, keep, input, n, float(a), float(b), float(dropped), BLOCK=1024
        )
    return input
