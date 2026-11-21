import torch
import triton
import triton.language as tl

# atleast_3d: aten::atleast_3d -- zero-copy rank promotion. rank 0 ->
# (1,1,1), rank 1 -> (1,N,1), rank 2 -> (M,N,1) via unsqueeze (view re-
# derivation of unit-axis strides would not match degenerate inputs);
# rank >= 3 returns the tensor itself (alias). The Sequence form applies
# the rule elementwise. The Triton copy kernel serves _atleast_3d_materialize
# and is never used by the aliasing paths.

_BLOCK_SIZE = 1024
_unsqueeze = torch.Tensor.unsqueeze
_Tensor = torch.Tensor


@triton.jit
def _atleast_3d_copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, values, mask=mask)


def _atleast_3d_materialize(x):
    """Fresh contiguous device copy of ``x`` in its promoted shape."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    n_elements = out.numel()
    if n_elements:
        _atleast_3d_copy_kernel[(triton.cdiv(n_elements, _BLOCK_SIZE),)](
            x, out, n_elements, BLOCK_SIZE=_BLOCK_SIZE
        )
    return out


def _tensor_3d(x):
    rank = x.ndim
    if rank == 2:
        return _unsqueeze(x, -1)
    if rank >= 3:
        return x
    if rank == 1:
        return _unsqueeze(_unsqueeze(x, 0), -1)
    return _unsqueeze(_unsqueeze(_unsqueeze(x, 0), 0), 0)


def _promote(item):
    if type(item) is _Tensor or isinstance(item, torch.Tensor):
        return _tensor_3d(item)
    raise TypeError("atleast_3d expects a Tensor inside a sequence")


def atleast_3d(self):
    """``aten::atleast_3d`` (and its Sequence form): rank-0/1/2 promoted views; else alias."""
    kind = type(self)
    if kind is _Tensor:
        return _tensor_3d(self)
    if kind is list or kind is tuple:
        return [_promote(item) for item in self]
    if isinstance(self, _Tensor):
        return _tensor_3d(self)
    if isinstance(self, (list, tuple)):
        return [_promote(item) for item in self]
    raise TypeError("atleast_3d expects a Tensor or a sequence of Tensors")
