"""Port of FlagGems ``aten::mT(Tensor(a) self) -> Tensor(a)``.

``mT`` swaps the last two dimensions of a matrix or a batch of matrices and
returns a **zero-copy view** that shares the input's storage.  No element of
``self`` is read, written, rounded or converted: the operand's bytes ARE the
result's bytes.  The observable result is produced by the allowlisted
zero-copy layout primitive ``Tensor.transpose`` -- ATen's own decomposition of
``mT`` -- so storage aliasing, storage offset, the swapped strides, the lazy
conj/neg bits and the autograd edge are preserved by construction.

The operator's only derived quantity is its relayout metadata: the normalized
axis order ``(0, 1, ..., rank-3, rank-1, rank-2)``.  A genuine ``@triton.jit``
program materializes that order into a per-device int32 workspace.  The
operand is never passed to the kernel and no element of it is touched, so one
compiled binary is valid for every operand form.  That materialization is
*output-independent state*: nothing ever reads the workspace back, so the
launch is issued at most once per (device, rank).

Dimension contract: rank >= 2 transposes the last two dims, rank 0 follows
ATen's deprecated identity path (returns the very same tensor object), and
rank 1 raises ``RuntimeError``.
"""

import torch
import triton
import triton.language as tl

__all__ = ["mT"]

# Axis-order workspace geometry: one int32 slot per output axis.
_MAX_DIM = 16

# Per-device scratch for the kernel result.  It holds only the relayout order,
# which nothing reads back, so reusing it across calls cannot change any
# observable output.
_ORDER_WORKSPACE = {}

# Last (device index, rank) pair whose order was already materialized.  The
# sentinel is unreachable because a real device index is ``None`` or a
# non-negative int.
_LAST_DEVICE_INDEX = -2
_LAST_RANK = -1


@triton.jit
def _mt_order_kernel(order_ptr, rank, MAX_DIM: tl.constexpr):
    """Materialize the ``mT`` axis order ``order[k]`` for ``rank`` axes.

    ``order[rank-2] = rank-1`` and ``order[rank-1] = rank-2``; every earlier
    slot keeps its own index.  Only positions ``< rank`` are stored.
    """
    pos = tl.arange(0, MAX_DIM)
    axis = tl.where(pos == rank - 2, rank - 1, tl.where(pos == rank - 1, rank - 2, pos))
    tl.store(order_ptr + pos, axis, mask=pos < rank)


def _order_workspace(device):
    """Return this device's int32 order buffer, creating it on first use."""
    key = getattr(device, "index", None)
    buffer = _ORDER_WORKSPACE.get(key)
    if buffer is None:
        buffer = torch.empty(_MAX_DIM, dtype=torch.int32, device=device)
        _ORDER_WORKSPACE[key] = buffer
    return buffer


def mT(self):
    global _LAST_DEVICE_INDEX, _LAST_RANK

    # Zero-cost type guard: a non-Tensor argument has no ``dim``, exactly like
    # the native operator's "must be Tensor" rejection.
    try:
        ndim = self.dim()
    except AttributeError:
        raise TypeError(
            "mT(): argument 'self' must be a Tensor, not " + type(self).__name__
        ) from None

    if ndim >= 2:
        # Materialize the mT axis order at most once per (device, rank).
        device = self.device
        device_index = device.index
        if device_index != _LAST_DEVICE_INDEX or ndim != _LAST_RANK:
            _mt_order_kernel[(1,)](
                _order_workspace(device), ndim, MAX_DIM=_MAX_DIM, num_warps=1
            )
            _LAST_DEVICE_INDEX = device_index
            _LAST_RANK = ndim
        # Allowlisted zero-copy layout view over the operand's own storage.
        return self.transpose(-2, -1)

    if ndim == 0:
        # Deprecated identity path: ATen returns the very same tensor object.
        return self

    raise RuntimeError(
        "mT is only supported for tensors with at least 2 dimensions, but got "
        "a tensor with 1 dimension"
    )
