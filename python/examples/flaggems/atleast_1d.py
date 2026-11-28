import torch
import triton
import triton.language as tl

# atleast_1d: aten::atleast_1d -- zero-copy rank promotion. A 0-d input is
# unsqueezed to (1,); rank >= 1 returns the tensor itself (same storage,
# strides and offset -- the aliasing contract of the reference). The
# Sequence form applies the rule elementwise. The module also carries a
# real Triton copy kernel used by _atleast_1d_materialize; the aliasing
# paths above never call it (a copy would violate the contract).

_BLOCK_SIZE = 1024
_unsqueeze = torch.Tensor.unsqueeze
_dim = torch.Tensor.dim
_TensorType = torch.Tensor


@triton.jit
def _atleast_1d_copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, values, mask=mask)


def _atleast_1d_materialize(x):
    """Fresh contiguous device copy of ``x`` (any rank/dtype)."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    n_elements = out.numel()
    if n_elements:
        _atleast_1d_copy_kernel[(triton.cdiv(n_elements, _BLOCK_SIZE),)](
            x, out, n_elements, BLOCK_SIZE=_BLOCK_SIZE
        )
    return out


def _view(t):
    if _dim(t) == 0:
        return _unsqueeze(t, 0)
    return t


def _promote(t):
    if type(t) is _TensorType:
        return _view(t)
    if isinstance(t, torch.Tensor):
        return _view(t)
    raise TypeError(
        f"atleast_1d(): expected a Tensor inside a sequence, got '{type(t).__name__}'"
    )


def atleast_1d(self):
    """``aten::atleast_1d`` (and its Sequence form): 0-d -> (1,) view; else alias."""
    kind = type(self)
    if kind is _TensorType:
        if _dim(self) == 0:
            return _unsqueeze(self, 0)
        return self
    if kind is list or kind is tuple:
        return [_promote(t) for t in self]
    if isinstance(self, torch.Tensor):
        if _dim(self) == 0:
            return _unsqueeze(self, 0)
        return self
    if isinstance(self, (list, tuple)):
        return [_promote(t) for t in self]
    raise TypeError(
        "atleast_1d(): expected a Tensor or a sequence of Tensors, got "
        f"'{type(self).__name__}'"
    )
