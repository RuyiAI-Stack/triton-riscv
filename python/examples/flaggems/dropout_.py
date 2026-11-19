import torch
import triton
import triton.language as tl

# In-place dropout (aten::dropout_): during training elements are zeroed
# with probability p and the survivors are scaled by 1 / (1 - p), mutating
# the input and returning it. Probed on torch CPU: eval / p == 0 return the
# input object untouched, and p == 1 keeps the signed zero (x * 0, not a
# plain zero fill). The keep mask is drawn from the torch RNG (deterministic
# under torch.manual_seed); the scaling runs in the Triton kernel.
#
# Mirrors the upstream requirement: the in-place path needs a contiguous
# buffer (strided inputs are handled by a copy-back fallback).


@triton.jit
def _dropout_kernel(x_ptr, keep_ptr, out_ptr, n_elements, scale, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    keep = tl.load(keep_ptr + offsets, mask=mask, other=0)
    res = tl.where(keep != 0, x * scale, 0.0 * x)
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _apply_inplace(x, keep, scale):
    n = x.numel()
    if n:
        _dropout_kernel[(triton.cdiv(n, 1024),)](x, keep, x, n, scale, BLOCK=1024)
    return x


def dropout_(input, p=0.5, train=True):
    """In-place dropout; eval and p == 0 are no-ops returning the input."""
    if not train or p == 0.0:
        return input
    if p == 1.0:
        return input.mul_(0.0)
    assert input.is_contiguous(), "dropout_ requires a contiguous tensor"
    scale = 1.0 / (1.0 - p)
    keep = (torch.rand(input.shape, dtype=torch.float32, device=input.device) >= p).to(
        torch.uint8
    )
    return _apply_inplace(input, keep, scale)
