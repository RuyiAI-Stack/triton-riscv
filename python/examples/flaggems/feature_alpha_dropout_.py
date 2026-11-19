import math

import torch
import triton
import triton.language as tl

# In-place feature_alpha_dropout: whole (sample, channel) channels drop
# together; the input is mutated and returned. eval / p == 0 are no-ops on
# the input object, p == 1 writes signed zeros.

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
def _feature_alpha_dropout_inplace_kernel(
    x_ptr,
    keep_ptr,
    out_ptr,
    n_elements,
    spatial,
    channels,
    a,
    b,
    dropped,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    channel = offsets // spatial
    keep = tl.load(keep_ptr + channel, mask=mask, other=0)
    res = tl.where(keep != 0, a * x + b, dropped)
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _launch(x, keep, a, b, dropped, out):
    n = x.numel()
    channels = x.size(1)
    spatial = n // (x.size(0) * channels) if n and channels else 0
    if n:
        _feature_alpha_dropout_inplace_kernel[(triton.cdiv(n, 1024),)](
            x,
            keep,
            out,
            n,
            spatial,
            channels,
            float(a),
            float(b),
            float(dropped),
            BLOCK=1024,
        )
    return out


def feature_alpha_dropout_(input, p=0.5, train=True):
    """In-place feature alpha dropout; eval and p == 0 are no-ops."""
    if not train or p == 0.0:
        return input
    if p == 1.0:
        return input.mul_(0.0)
    a, b = _alpha_dropout_affine(p)
    dropped = a * _ALPHA_P + b
    assert input.is_contiguous(), "feature_alpha_dropout_ requires a contiguous tensor"
    keep = (torch.rand(input.shape[0] * input.shape[1]) >= p).to(torch.uint8)
    return _launch(input, keep, a, b, dropped, input)
