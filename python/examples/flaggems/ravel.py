import torch
import triton
import triton.language as tl

# ravel: aten::ravel -- flatten to 1-D. Contiguous inputs return a zero-copy
# view (the reference contract); anything else is materialized by a Triton
# gather into a fresh flat tensor. An autograd node keeps the view/copy
# distinction transparent for gradients (the backward is a pure reshape).


@triton.jit
def _ravel_gather_kernel(
    src,
    dst,
    numel,
    SHAPE: tl.constexpr,
    INPUT_STRIDES: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    rem = offs.to(tl.int64)
    src_off = tl.zeros([BLOCK], dtype=tl.int64)
    for d in tl.static_range(NDIM - 1, -1, -1):
        idx = rem % SHAPE[d]
        rem = rem // SHAPE[d]
        src_off += idx * INPUT_STRIDES[d]
    vals = tl.load(src + src_off, mask=mask)
    tl.store(dst + offs, vals, mask=mask)


def _copy_impl(x):
    out = torch.empty(x.numel(), dtype=x.dtype, device=x.device)
    if x.is_complex():
        src = torch.view_as_real(x)
        dst = torch.view_as_real(out.view(x.shape)).reshape(-1)
        numel = dst.numel()
    else:
        src = x
        dst = out
        numel = out.numel()
    if numel > 0:
        grid = (triton.cdiv(numel, 1024),)
        _ravel_gather_kernel[grid](
            src,
            dst,
            numel,
            SHAPE=tuple(src.shape),
            INPUT_STRIDES=tuple(src.stride()),
            NDIM=src.dim(),
            BLOCK=1024,
        )
    return out


class _RavelCopy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.in_shape = tuple(x.shape)
        return _copy_impl(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.view(ctx.in_shape)


def ravel(self):
    """Flatten to 1-D: a view when contiguous (reference contract), else a fresh copy."""
    if not isinstance(self, torch.Tensor):
        raise TypeError(
            "ravel(): argument 'self' must be a Tensor, not %s" % type(self).__name__
        )
    if self.layout != torch.strided:
        raise RuntimeError("unsupported memory format option Contiguous")
    if self.is_contiguous():
        return self.view(-1)
    if self.requires_grad and torch.is_grad_enabled():
        return _RavelCopy.apply(self)
    return _copy_impl(self)
