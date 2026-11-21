import torch
import triton
import triton.language as tl

# _new_zeros_with_same_feature_meta: aten's nested-tensor zero allocator.
# Returns a zero-filled tensor whose feature dims (trailing, after
# ``self_num_batch_dims`` batch dims) reproduce ``other``'s shape, strides
# and storage offset layout. Zero stores go through Triton: flat masked
# stores for contiguous views, a meta-driven strided store otherwise;
# complex tensors zero via their real views, bool via uint8.

_BLOCK = 1024
_SBLOCK = 256


@triton.jit
def _zero_fill_contig(ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offs, tl.zeros([BLOCK], dtype=ptr.dtype.element_ty), mask=offs < n)


@triton.jit
def _zero_fill_strided(ptr, meta, n, RANK: tl.constexpr, BLOCK: tl.constexpr):
    # ``meta`` holds the sizes followed by the strides, both int64.
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx
    off = tl.zeros([BLOCK], dtype=tl.int64)
    for d in tl.static_range(RANK):
        size = tl.load(meta + d)
        step = tl.load(meta + RANK + d)
        off += (rem % size) * step
        rem = rem // size
    tl.store(ptr + off, tl.zeros([BLOCK], dtype=ptr.dtype.element_ty), mask=mask)


def _fill(t):
    n = t.numel()
    if n == 0:
        return t
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    if t.is_contiguous():
        _zero_fill_contig[grid](t, n, BLOCK=_BLOCK)
    else:
        sizes = t.shape
        strides = t.stride()
        rank = len(sizes)
        meta = torch.empty(2 * rank, dtype=torch.int64, device=t.device)
        meta[:rank] = torch.tensor(sizes, dtype=torch.int64, device=t.device)
        meta[rank:] = torch.tensor(strides, dtype=torch.int64, device=t.device)
        grid = ((n + _SBLOCK - 1) // _SBLOCK,)
        _zero_fill_strided[grid](t, meta, n, RANK=rank, BLOCK=_SBLOCK)
    return t


def _zero(t):
    # The reinterpreted view aliases ``t``, so ``t`` itself is always returned.
    if t.is_complex():
        _fill(torch.view_as_real(t))
    elif t.dtype == torch.bool:
        _fill(t.view(torch.uint8))
    else:
        _fill(t)
    return t


def _storage_numel(t):
    try:
        return t.untyped_storage().nbytes() // t.element_size()
    except RuntimeError:
        return 0


def _empty_strides(shape):
    # Strides ``torch.empty(shape)`` would report (zero dims count as 1).
    strides = [0] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        if shape[i] != 0:
            acc *= shape[i]
    return tuple(strides)


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _new_zeros_with_same_feature_meta(self, other, *, self_num_batch_dims=0):
    """Zero tensor with ``self``'s batch dims and ``other``'s feature shape/strides."""
    if not (isinstance(self, torch.Tensor) and isinstance(other, torch.Tensor)):
        raise TypeError("self and other must be torch.Tensor")
    if isinstance(self_num_batch_dims, bool) or not isinstance(
        self_num_batch_dims, int
    ):
        raise TypeError(
            "self_num_batch_dims must be an int, got %r" % type(self_num_batch_dims)
        )
    n = self_num_batch_dims
    if n < 0:
        raise RuntimeError("mismatch in length of strides and shape")
    self_dim = self.dim()
    batch = (
        tuple(self.shape[:n])
        if n <= self_dim
        else tuple(self.shape) + (0,) * (n - self_dim)
    )
    # Default form: fresh storage that reproduces `other`'s feature meta.
    other_shape = tuple(other.shape)
    other_stride = other.stride()
    offset = other.storage_offset()
    snum = _storage_numel(other)
    src_strides = [0] * len(batch)
    base = snum
    for i in range(len(batch) - 1, -1, -1):
        src_strides[i] = base
        base *= batch[i]
    src_strides += list(other_stride)
    shape = batch + other_shape
    # Common case (non-empty shape, contiguous `other` with no storage
    # offset): the derived layout is torch's contiguous layout, so the plain
    # allocation is exact.  Shapes containing a zero-sized dim are excluded
    # because torch's own stride convention for them is not the ATen rule
    # the buffer path reproduces.
    if offset == 0 and 0 not in shape and tuple(src_strides) == _empty_strides(shape):
        return _zero(torch.empty(shape, dtype=other.dtype, device=other.device))
    need = _numel(batch) * snum
    buf = torch.empty(need, dtype=other.dtype, device=other.device)
    return _zero(buf.as_strided(shape, src_strides, storage_offset=offset))
