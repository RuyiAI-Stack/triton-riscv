import math

import torch
import triton
import triton.language as tl

# feature_alpha_dropout: like alpha_dropout but whole feature channels
# (dim 1 slices) are dropped together. Probed on torch CPU: the mask
# granularity is per (sample, channel) -- each (n, c) channel is either
# fully dropped (constant a * alpha' + b) or fully kept (a * x + b), and
# different samples of the same channel drop independently. eval is a
# pass-through returning the input object.

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
def _feature_alpha_dropout_kernel(
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
    # (n, c) channel index of this element: same decode as the prelu port.
    channel = offsets // spatial
    keep = tl.load(keep_ptr + channel, mask=mask, other=0)
    res = tl.where(keep != 0, a * x + b, dropped)
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _apply(x, keep_ch, a, b, dropped):
    out = torch.empty_like(x)
    n = x.numel()
    channels = x.size(1)
    spatial = n // (x.size(0) * channels) if n and channels else 0
    if n:
        _feature_alpha_dropout_kernel[(triton.cdiv(n, 1024),)](
            x,
            keep_ch,
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


def feature_alpha_dropout(input, p=0.5, train=True):
    if not train or p == 0.0:
        return input
    if p == 1.0:
        return input * 0.0
    a, b = _alpha_dropout_affine(p)
    dropped = a * _ALPHA_P + b
    x = input.contiguous()
    # One Bernoulli draw per (sample, channel). The kernel decodes the flat
    # element offset as (n * C + c) * spatial + s, so a flat (N*C,) draw in
    # the same n*C + c order indexes the channel keep mask directly.
    keep = (torch.rand(x.shape[0] * x.shape[1]) >= p).to(torch.uint8)
    return _apply(x, keep, a, b, dropped)
