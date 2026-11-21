import logging

import torch
import triton
import triton.language as tl

# diagflat: aten::diagflat via a Triton zero-fill plus a strided diagonal
# scatter (and the matching gather for the autograd backward). Upstream's two
# fused single-launch fast paths are dropped: one branched on runtime scalar
# conditions inside the kernel (aborts codegen on this backend) and the other
# relied on tl.debug_barrier ordering; the separate fill+scatter launches
# produce identical values.
#
# Deviations from upstream, pinned to the aten oracle (torch 2.15.0+cpu):
# - argument TypeErrors use aten's exact text ("argument 'input' (position 1)
#   must be Tensor, not ..." / "argument 'offset' (position 2) must be int,
#   not ..."); upstream named the first argument 'self'.
# - a non-contiguous complex input fell into NotImplementedError; aten
#   computes it fine, so the flat source is materialized with .contiguous()
#   first (values identical).
# - bool tensors are scattered through a uint8 view (bit-identical), the
#   repo convention for 1-byte Triton storage.

logger = logging.getLogger(__name__)

_BLOCK_FILL = 4096
_BLOCK_SCATTER = 512
_INT32_MAX = 2147483647

_COMPLEX_REAL = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


def _fill_view(itemsize):
    """A same-or-larger-itemsize integer view type used for the zero fill."""
    if itemsize >= 8:
        return torch.int64
    if itemsize == 4:
        return torch.int32
    if itemsize == 2:
        return torch.int16
    return torch.uint8


@triton.jit(do_not_specialize=["count"])
def _diagflat_zero_kernel(out_ptr, count, BLOCK: tl.constexpr, WIDE: tl.constexpr):
    """Masked zero fill of the raw output buffer."""
    pid = tl.program_id(0)
    if WIDE:
        base = pid.to(tl.int64) * BLOCK
    else:
        base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    zero = tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, zero, mask=offs < count)


@triton.jit(do_not_specialize=["count", "sstride", "dstride", "base", "jmul"])
def _diagflat_scatter_kernel(
    src_ptr, dst_ptr, count, sstride, dstride, base, jmul, BLOCK: tl.constexpr
):
    """dst[i * dstride + base - (i & 1) * jmul] = src[i * sstride]"""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < count
    val = tl.load(
        src_ptr + offs * sstride,
        mask=mask,
        other=tl.zeros([BLOCK], dtype=src_ptr.dtype.element_ty),
    )
    tl.store(dst_ptr + (offs * dstride + base - (offs & 1) * jmul), val, mask=mask)


@triton.jit(do_not_specialize=["count", "sstride", "base", "jmul"])
def _diagflat_gather_kernel(
    src_ptr, dst_ptr, count, sstride, base, jmul, BLOCK: tl.constexpr
):
    """dst[i] = src[i * sstride + base - (i & 1) * jmul]"""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < count
    val = tl.load(
        src_ptr + (offs * sstride + base - (offs & 1) * jmul),
        mask=mask,
        other=tl.zeros([BLOCK], dtype=src_ptr.dtype.element_ty),
    )
    tl.store(dst_ptr + offs, val, mask=mask)


def _zero_fill(out):
    buf = out.view(_fill_view(out.element_size()))
    count = buf.numel()
    if count:
        grid = (triton.cdiv(count, _BLOCK_FILL),)
        _diagflat_zero_kernel[grid](
            buf,
            count,
            BLOCK=_BLOCK_FILL,
            WIDE=(count + 2 * _BLOCK_FILL) > _INT32_MAX,
            num_warps=4,
        )


def _diagonal_params(out, k):
    """``(base, jmul, real_dtype)`` for the offset-k diagonal of ``out``.

    For real dtypes the target address is ``i * (m + 1) + base`` with
    ``jmul == 0``. Complex tensors are viewed as their real component type
    with a doubled inner extent, giving ``i * (m + 1) + base - (i & 1) * m``.
    """
    m = out.shape[0]
    if out.is_complex():
        base = 2 * k if k >= 0 else 2 * (-k) * m
        return base, m, _COMPLEX_REAL[out.dtype]
    base = k if k >= 0 else (-k) * m
    return base, 0, None


def _forward(self, k):
    n = self.numel()
    m = n + (k if k >= 0 else -k)
    out = torch.empty((m, m), dtype=self.dtype, device=self.device)
    if out.numel() == 0:
        return out

    _zero_fill(out)

    if n > 0:
        flat = self.reshape(-1)
        base, jmul, real_dtype = _diagonal_params(out, k)
        if real_dtype is not None:
            if not flat.is_contiguous():
                flat = flat.contiguous()
            src = flat.view(real_dtype)
            dst = out.view(real_dtype)
            count = 2 * n
            sstride = 1
        else:
            src = flat
            dst = out
            count = n
            sstride = flat.stride(0)
            if src.dtype == torch.bool:
                src = src.view(torch.uint8)
                dst = out.view(torch.uint8)
        grid = (triton.cdiv(count, _BLOCK_SCATTER),)
        _diagflat_scatter_kernel[grid](
            src,
            dst,
            count,
            sstride,
            m + 1,
            base,
            jmul,
            BLOCK=_BLOCK_SCATTER,
            num_warps=4,
        )
    return out


def _backward(grad_out, shape, k):
    m = grad_out.shape[0]
    n = m - (k if k >= 0 else -k)
    grad_in = torch.empty(shape, dtype=grad_out.dtype, device=grad_out.device)
    if n <= 0 or grad_in.numel() == 0:
        return grad_in
    grad_out = grad_out.contiguous()

    if grad_out.is_complex():
        real_dtype = _COMPLEX_REAL[grad_out.dtype]
        src = grad_out.view(real_dtype)
        dst = grad_in.view(real_dtype)
        count = 2 * n
        sstride = m + 1
        base = 2 * k if k >= 0 else 2 * (-k) * m
        jmul = m
    else:
        src = grad_out
        dst = grad_in
        count = n
        sstride = m + 1
        base = k if k >= 0 else (-k) * m
        jmul = 0

    grid = (triton.cdiv(count, _BLOCK_SCATTER),)
    _diagflat_gather_kernel[grid](
        src, dst, count, sstride, base, jmul, BLOCK=_BLOCK_SCATTER, num_warps=4
    )
    return grad_in


class _DiagFlat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, offset):
        ctx.input_shape = self.shape
        ctx.offset = offset
        return _forward(self, offset)

    @staticmethod
    def backward(ctx, grad_out):
        return _backward(grad_out, ctx.input_shape, ctx.offset), None


def diagflat(self, offset=0):
    """``torch.diagflat``: flatten ``self`` onto the offset-k diagonal of a zero matrix."""
    if not isinstance(self, torch.Tensor):
        raise TypeError(
            f"diagflat(): argument 'input' (position 1) must be Tensor, "
            f"not {type(self).__name__}"
        )
    if not isinstance(offset, int):
        raise TypeError(
            f"diagflat(): argument 'offset' (position 2) must be int, "
            f"not {type(offset).__name__}"
        )
    k = int(offset)
    if self.requires_grad:
        return _DiagFlat.apply(self, k)
    return _forward(self, k)
