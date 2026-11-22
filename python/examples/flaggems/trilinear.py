"""Triton implementation of aten::_trilinear.

Ported from upstream FlagGems ops/trilinear.py.

Inserts a size-1 dimension at each position listed in ``expand1``/``expand2``/
``expand3`` (aten unsqueeze semantics), broadcasts the three operands to one
shape, multiplies them elementwise on-device, and sums over ``sumdim``.
``unroll_dim`` does not affect the mathematical result and is accepted for
aten signature compatibility.

Porting notes (repo conventions):
  - @libentry/@libtuner stripped (plain @triton.jit); the tuner's three
    configs collapse to the fixed BLOCK_SIZE=256; no torch device context
    manager; the grid is a plain tuple.
  - The FlagGems sum_dim/copy_/contiguous helpers are inlined: the sum runs
    through ``_trilinear_sum_kernel`` (one program per output row, the
    contracted axes permuted to the end), layout normalization uses plain
    ``Tensor.contiguous()`` and the out variant writes through
    ``Tensor.copy_``.
  - ``sumdim`` defaults to no reduction: the pinned tests call the op
    without it when no axes are summed.
"""

import warnings

import torch
import triton
import triton.language as tl


@triton.jit
def trilinear_kernel(
    i1_ptr,
    i2_ptr,
    i3_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    i1 = tl.load(i1_ptr + offsets, mask=mask, other=0.0)
    i2 = tl.load(i2_ptr + offsets, mask=mask, other=0.0)
    i3 = tl.load(i3_ptr + offsets, mask=mask, other=0.0)
    out = i1 * i2 * i3
    tl.store(output_ptr + offsets, out, mask=mask)


@triton.jit
def _trilinear_sum_kernel(
    in_ptr,
    out_ptr,
    inner,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce one output row: sum the ``inner`` contracted elements of row
    ``program_id(0)``, accumulated in the output dtype."""
    row = tl.program_id(0)
    base = in_ptr + row.to(tl.int64) * inner
    acc = tl.zeros((BLOCK_SIZE,), dtype=out_ptr.dtype.element_ty)
    for start in range(0, inner, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        acc += tl.load(base + offs, mask=offs < inner, other=0)
    tl.store(out_ptr + row, tl.sum(acc, axis=0))


def _apply_expand(t, expand):
    """Insert size-1 dims at the positions listed in ``expand`` (aten semantics)."""
    for d in expand:
        t = t.unsqueeze(d)
    return t


def _trilinear_product(i1, i2, i3, expand1, expand2, expand3):
    """Broadcast the three inputs and compute their elementwise product on-device
    with the Triton ``trilinear_kernel`` (no torch elementwise-mul fallback)."""
    a = _apply_expand(i1, expand1)
    b = _apply_expand(i2, expand2)
    c = _apply_expand(i3, expand3)

    shape = torch.broadcast_shapes(a.shape, b.shape, c.shape)
    a = a.broadcast_to(shape)
    b = b.broadcast_to(shape)
    c = c.broadcast_to(shape)
    a = a.contiguous() if not a.is_contiguous() else a
    b = b.contiguous() if not b.is_contiguous() else b
    c = c.contiguous() if not c.is_contiguous() else c

    prod = torch.empty(shape, dtype=a.dtype, device=a.device)
    n_elements = prod.numel()
    if n_elements > 0:
        grid = (triton.cdiv(n_elements, 256),)
        trilinear_kernel[grid](a, b, c, prod, n_elements, BLOCK_SIZE=256)
    return prod


def _trilinear_reduce(prod, sumdim):
    """Sum ``prod`` over ``sumdim`` with the inline reduction kernel."""
    free = [d for d in range(prod.dim()) if d not in sumdim]
    t = prod.permute(free + list(sumdim))
    outer = 1
    for d in free:
        outer *= prod.shape[d]
    inner = prod.numel() // outer if outer else 0
    out_shape = tuple(prod.shape[d] for d in free)

    out = torch.empty(out_shape, dtype=prod.dtype, device=prod.device)
    if outer > 0:
        rows = t.reshape(outer, inner)
        block = max(min(triton.next_power_of_2(inner), 256), 2)
        _trilinear_sum_kernel[(outer,)](rows, out, inner, BLOCK_SIZE=block)
    return out


def _trilinear_compute(i1, i2, i3, expand1, expand2, expand3, sumdim):
    prod = _trilinear_product(i1, i2, i3, expand1, expand2, expand3)
    if sumdim:
        prod = _trilinear_reduce(prod, list(sumdim))
    return prod


def _trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim=None, unroll_dim=1):
    """Compute the trilinear product exp(A)-style 3-way tensor product.

    Broadcasts ``i1``/``i2``/``i3`` (after applying the ``expand*`` unsqueeze
    lists) to a common shape, multiplies them elementwise with a Triton kernel,
    and sums over ``sumdim``. ``unroll_dim`` does not affect the mathematical
    result and is accepted for aten signature compatibility.
    """
    if not (
        isinstance(i1, torch.Tensor)
        and isinstance(i2, torch.Tensor)
        and isinstance(i3, torch.Tensor)
    ):
        raise TypeError("_trilinear expects three tensor inputs")
    if not (i1.device == i2.device == i3.device):
        raise RuntimeError("All tensors must be on the same device")
    return _trilinear_compute(i1, i2, i3, expand1, expand2, expand3, sumdim)


def _trilinear_out(
    i1, i2, i3, expand1, expand2, expand3, sumdim=None, unroll_dim=1, *, out
):
    """Out variant of :func:`_trilinear` writing the result into ``out``."""
    if not (
        isinstance(i1, torch.Tensor)
        and isinstance(i2, torch.Tensor)
        and isinstance(i3, torch.Tensor)
    ):
        raise TypeError("_trilinear expects three tensor inputs")
    if not (i1.device == i2.device == i3.device):
        raise RuntimeError("All tensors must be on the same device")
    if out.device != i1.device:
        raise RuntimeError("output must be on the same device as inputs")
    result = _trilinear_compute(i1, i2, i3, expand1, expand2, expand3, sumdim)
    # ATen out contract: resize when the shape differs (matching aten's
    # resize_output semantics), then write through the out tensor's own strides.
    if tuple(out.shape) != tuple(result.shape):
        if out.numel() != 0:
            warnings.warn(
                "An output with one or more elements was resized since it had "
                f"shape {list(out.shape)}, which does not match the required "
                f"output shape {list(result.shape)}. This behavior is deprecated, "
                "and in a future PyTorch release outputs will not be resized "
                "unless they have zero elements. You can explicitly reuse an out "
                "tensor t by resizing it, inplace, to zero elements with "
                "t.resize_(0).",
                UserWarning,
                stacklevel=3,
            )
        out.resize_(result.shape)
    out.copy_(result)
    return out
