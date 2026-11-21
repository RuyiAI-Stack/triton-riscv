import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["pad_left", "pad_top"])
def _replication_pad2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    in_h,
    in_w,
    out_h,
    out_w,
    pad_left,
    pad_top,
    in_stride_nc,
    out_stride_nc,
    BLOCK_W: tl.constexpr,
):
    """Input-space gather for the replication_pad2d backward.

    Each program owns one input row (nc, ih) and a BLOCK_W tile of input
    columns. It sweeps the whole grad_output in ascending (oh, ow) order and
    adds, per input lane, every output element whose replicated source is
    (ih, iw). Adds therefore run in aten's exact CPU order and in the element
    dtype, making results bit-exact against aten (fp64 included) with no
    atomics. Upstream used an interior copy plus an atomic edge scatter in
    float32, which is neither deterministic nor fp64-exact.
    """
    pid_row = tl.program_id(0)
    pid_w = tl.program_id(1)

    ih = pid_row % in_h
    nc = pid_row // in_h

    iw = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = iw < in_w

    grad_base = grad_output_ptr + nc * out_stride_nc

    # Scalar (oh, ow) sweep: each scalar grad value is broadcast onto the iw
    # lanes whose replicated source it hits. Per input lane the adds then run
    # in ascending (oh, ow) order -- aten's exact accumulation order -- in the
    # element dtype.
    acc = tl.zeros((BLOCK_W,), dtype=grad_output_ptr.dtype.element_ty)
    for oh in range(0, out_h):
        target_ih = tl.minimum(tl.maximum(oh - pad_top, 0), in_h - 1)
        row_hit = target_ih == ih
        for ow in range(0, out_w):
            target_iw = tl.minimum(tl.maximum(ow - pad_left, 0), in_w - 1)
            hit = mask & row_hit & (target_iw == iw)
            g = tl.load(grad_base + oh * out_w + ow)
            acc += tl.where(hit, g, 0.0)

    grad_in_base = grad_input_ptr + nc * in_stride_nc + ih * in_w
    tl.store(
        grad_in_base + iw,
        acc.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


def _replication_pad2d_backward_impl(
    grad_output: torch.Tensor, self: torch.Tensor, padding, *, out: torch.Tensor = None
) -> torch.Tensor:
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif isinstance(padding, (tuple, list)):
        if len(padding) != 4:
            raise ValueError(
                "padding must be a sequence of 4 integers: "
                "(pad_left, pad_right, pad_top, pad_bottom)"
            )
        pad_left, pad_right, pad_top, pad_bottom = map(int, padding)
    else:
        raise TypeError(f"Unexpected padding type: {type(padding)}")

    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("Padding values must be non-negative")

    is_3d = self.ndim == 3
    if is_3d:
        C, H, W = self.shape
        N = 1
        grad_output_4d = grad_output.view(
            1, C, grad_output.shape[-2], grad_output.shape[-1]
        )
    elif self.ndim == 4:
        N, C, H, W = self.shape
        grad_output_4d = grad_output.contiguous()
    else:
        raise ValueError(
            "replication_pad2d_backward expects a 3D (C, H, W) or 4D (N, C, H, W) input"
        )

    OH = H + pad_top + pad_bottom
    OW = W + pad_left + pad_right

    expected_grad_shape = (N, C, OH, OW)
    if tuple(grad_output_4d.shape) != expected_grad_shape:
        raise ValueError(
            f"grad_output shape {tuple(grad_output_4d.shape)} does not match "
            f"expected shape {expected_grad_shape}"
        )

    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        result = grad_output_4d.contiguous().to(self.dtype)
        if is_3d:
            result = result.view(C, H, W)
        if out is not None:
            out.copy_(result)
            return out
        return result

    nc_total = N * C

    if out is not None:
        accum = out
    else:
        accum = torch.empty((N, C, H, W), device=self.device, dtype=self.dtype)

    if accum.numel() > 0:
        grad_output_4d = grad_output_4d.contiguous()
        BLOCK_W = 256
        grid = (nc_total * H, triton.cdiv(W, BLOCK_W))
        _replication_pad2d_backward_kernel[grid](
            grad_output_4d,
            accum,
            H,
            W,
            OH,
            OW,
            pad_left,
            pad_top,
            H * W,
            OH * OW,
            BLOCK_W=BLOCK_W,
        )

    if is_3d:
        accum = accum.view(C, H, W)
    return accum


def replication_pad2d_backward(
    grad_output: torch.Tensor, self: torch.Tensor, padding
) -> torch.Tensor:
    """aten::replication_pad2d_backward; padding order (l, r, t, b) like aten.

    Contributions accumulate per input cell in ascending output order in the
    element dtype (aten's CPU semantics), bit-exact for fp64/fp32/fp16.
    """
    logger.debug("GEMS REPLICATION_PAD2D_BACKWARD")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=None)


def replication_pad2d_backward_grad_input(
    grad_output: torch.Tensor, self: torch.Tensor, padding, *, grad_input: torch.Tensor
) -> torch.Tensor:
    """``replication_pad2d_backward`` writing into ``grad_input`` (aten's
    ``.grad_input`` overload)."""
    logger.debug("GEMS REPLICATION_PAD2D_BACKWARD_GRAD_INPUT")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=grad_input)
