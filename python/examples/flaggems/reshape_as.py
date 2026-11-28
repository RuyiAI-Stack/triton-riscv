import torch
import triton
import triton.language as tl

# reshape_as: aten::reshape_as -- reshape to other's shape. A stride-
# expressible target returns a zero-copy view; anything else materializes
# through a Triton gather from self's strides (complex inputs gather
# through their real-pair views). An autograd node keeps gradients
# flowing; its backward is a pure reshape.


@triton.jit
def _reshape_as_gather_kernel(
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


def _copy_logical(src, dst_flat):
    # dst_flat is a contiguous flat view of the output; src may be strided.
    if src.is_complex():
        gather_src = torch.view_as_real(src)
        numel = gather_src.numel()
        gather_dst = torch.view_as_real(dst_flat.view(src.shape)).reshape(-1)
    else:
        gather_src = src
        gather_dst = dst_flat
        numel = dst_flat.numel()
    if numel > 0:
        grid = (triton.cdiv(numel, 1024),)
        _reshape_as_gather_kernel[grid](
            gather_src,
            gather_dst,
            numel,
            SHAPE=tuple(gather_src.shape),
            INPUT_STRIDES=tuple(gather_src.stride()),
            NDIM=gather_src.dim(),
            BLOCK=1024,
        )


class _ReshapeAsMaterialize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, src_shape, target_shape):
        out = torch.empty(target_shape, dtype=src.dtype, device=src.device)
        _copy_logical(src, out.reshape(-1))
        ctx.in_shape = src_shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.reshape(ctx.in_shape), None, None


def reshape_as(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """``self`` reshaped to ``other``'s shape; a view when viewable, else a copy."""
    if not isinstance(self, torch.Tensor):
        raise TypeError("reshape_as expects a torch.Tensor input for 'self'")
    if not isinstance(other, torch.Tensor):
        raise TypeError("reshape_as expects a torch.Tensor input for 'other'")
    target_shape = tuple(other.size())
    if self.numel() != other.numel():
        raise RuntimeError(
            f"shape '{list(target_shape)}' is invalid for input of size {self.numel()}"
        )
    # A layout the view can express returns a zero-copy view, matching
    # native reshape's computeStride decision.
    try:
        return self.view(target_shape)
    except RuntimeError:
        pass
    if self.requires_grad and torch.is_grad_enabled():
        return _ReshapeAsMaterialize.apply(self, tuple(self.shape), target_shape)
    out = torch.empty(target_shape, dtype=self.dtype, device=self.device)
    _copy_logical(self, out.reshape(-1))
    return out
