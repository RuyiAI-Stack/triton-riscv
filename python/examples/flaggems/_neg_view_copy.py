import torch
import triton
import triton.language as tl

# _neg_view_copy: aten::_neg_view_copy -- a fresh contiguous tensor holding
# -self. Floating dtypes flip the sign via v * (-1.0) so -0.0 stays
# IEEE-visible; integer dtypes use two's-complement unary minus. A
# joint-dimension collapse feeds the strided kernel; the flat kernel takes
# an EVEN fast path. Wrapped in an autograd node (d(-x)/dx == -grad_out).

_SUPPORTED = frozenset(
    (
        torch.float64,
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
    )
)
_STRIDED_BLOCK = 1024
_MAX_RANK = 8


@triton.jit
def _neg_flat_kernel(
    in_ptr,
    out_ptr,
    numel,
    IS_FLOAT: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        v = tl.load(in_ptr + offs)
        if IS_FLOAT:
            tl.store(out_ptr + offs, v * (-1.0))
        else:
            tl.store(out_ptr + offs, -v)
    else:
        m = offs < numel
        v = tl.load(in_ptr + offs, mask=m)
        if IS_FLOAT:
            tl.store(out_ptr + offs, v * (-1.0), mask=m)
        else:
            tl.store(out_ptr + offs, -v, mask=m)


_NS_ARGS = [
    "d0",
    "d1",
    "d2",
    "d3",
    "d4",
    "d5",
    "d6",
    "d7",
    "s0",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
    "t0",
    "t1",
    "t2",
    "t3",
    "t4",
    "t5",
    "t6",
    "t7",
]


@triton.jit(do_not_specialize=_NS_ARGS)
def _neg_strided_kernel(
    in_ptr,
    out_ptr,
    numel,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    IS_FLOAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Padded 8-slot static decode (size-1/stride-0 slots contribute
    # nothing); the upstream chained runtime `if dN > 1` branches abort the
    # CPU backend's codegen, so the decode is the static_range form used by
    # the sibling copy kernels.
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < numel
    dims = [d0, d1, d2, d3, d4, d5, d6, d7]
    in_strides = [s0, s1, s2, s3, s4, s5, s6, s7]
    out_strides = [t0, t1, t2, t3, t4, t5, t6, t7]
    in_off = tl.zeros([BLOCK], dtype=tl.int64)
    out_off = tl.zeros([BLOCK], dtype=tl.int64)
    rem = offs
    for j in tl.static_range(8 - 1, -1, -1):
        idx = rem % dims[j]
        rem = rem // dims[j]
        in_off += idx * in_strides[j]
        out_off += idx * out_strides[j]
    v = tl.load(in_ptr + in_off, mask=m)
    if IS_FLOAT:
        tl.store(out_ptr + out_off, v * (-1.0), mask=m)
    else:
        tl.store(out_ptr + out_off, -v, mask=m)


def _joint_collapse(shape, in_strides, out_strides):
    # Merge adjacent dims that are contiguous for input *and* output; the
    # logical multi-index is unchanged by the merge.
    merged = []
    for size, ins, outs in zip(shape, in_strides, out_strides):
        if size == 1:
            continue
        if merged:
            prev_size, prev_ins, prev_outs = merged[-1]
            if prev_ins == ins * size and prev_outs == outs * size:
                merged[-1] = (prev_size * size, ins, outs)
                continue
        merged.append((size, ins, outs))
    return merged


def _launch_strided(src, dst, numel):
    dims = _joint_collapse(src.shape, src.stride(), dst.stride())
    if len(dims) > _MAX_RANK:
        raise RuntimeError(
            "_neg_view_copy: tensor rank too large for the strided kernel"
        )
    sizes = [1] * _MAX_RANK
    in_strides = [0] * _MAX_RANK
    out_strides = [0] * _MAX_RANK
    for k, (size, ins, outs) in enumerate(dims):
        sizes[k] = size
        in_strides[k] = ins
        out_strides[k] = outs
    grid = ((numel + _STRIDED_BLOCK - 1) // _STRIDED_BLOCK,)
    _neg_strided_kernel[grid](
        src,
        dst,
        numel,
        *sizes,
        *in_strides,
        *out_strides,
        IS_FLOAT=1 if src.dtype.is_floating_point else 0,
        BLOCK=_STRIDED_BLOCK,
    )


def _negate(src, dst):
    numel = src.numel()
    if numel == 0:
        return dst
    if src.is_contiguous() and dst.is_contiguous():
        block = 1024
        even = numel % block == 0
        grid = ((numel + block - 1) // block,)
        _neg_flat_kernel[grid](
            src,
            dst,
            numel,
            IS_FLOAT=1 if src.dtype.is_floating_point else 0,
            EVEN=1 if even else 0,
            BLOCK=block,
        )
    else:
        _launch_strided(src, dst, numel)
    return dst


class _NegViewCopy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        return _negate(x, out)

    @staticmethod
    def backward(ctx, grad_out):
        out = torch.empty(grad_out.shape, dtype=grad_out.dtype, device=grad_out.device)
        return _negate(grad_out, out)


def _neg_view_copy(self: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """Fresh contiguous tensor holding ``-self`` (``out=`` selects the .out overload)."""
    if out is not None:
        return _neg_view_copy_out(self, out=out)
    if not isinstance(self, torch.Tensor):
        raise TypeError("_neg_view_copy expects a torch.Tensor input")
    if self.dtype not in _SUPPORTED:
        raise RuntimeError("_neg_view_copy is not implemented for " + str(self.dtype))
    if self.requires_grad and torch.is_grad_enabled():
        return _NegViewCopy.apply(self)
    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    return _negate(self, out)


def _neg_view_copy_out(self: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """Write ``-self`` into ``out`` and return it."""
    if not isinstance(self, torch.Tensor):
        raise TypeError("_neg_view_copy expects a torch.Tensor input")
    if self.dtype not in _SUPPORTED:
        raise RuntimeError("_neg_view_copy is not implemented for " + str(self.dtype))
    if not isinstance(out, torch.Tensor):
        raise TypeError("_neg_view_copy.out expects a torch.Tensor out buffer")
    if out.dtype != self.dtype:
        raise RuntimeError("_neg_view_copy.out: dtype of input and output must match.")
    if out.device != self.device:
        raise RuntimeError(
            "_neg_view_copy.out: input and output must be on the same device."
        )
    if out.shape != self.shape:
        out.resize_(self.shape)
    return _negate(self, out)
