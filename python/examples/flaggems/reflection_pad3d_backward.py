import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def reflection_pad3d_backward_kernel(
    grad_ptr,
    out_ptr,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    pad_d0,
    pad_d1,
    pad_h0,
    pad_h1,
    pad_w0,
    pad_w1,
    stride_grad_n,
    stride_grad_c,
    stride_grad_d,
    stride_grad_h,
    stride_grad_w,
    stride_out_n,
    stride_out_c,
    stride_out_d,
    stride_out_h,
    stride_out_w,
    BLOCK_DHW: tl.constexpr,
):
    """Input-space gather for the reflection_pad3d backward.

    Each input voxel collects up to 27 gradient contributions (3 candidate
    output coordinates per axis: front/top/left reflection, center,
    back/bottom/right reflection). The adds run in ascending output order
    (od outer, oh middle, ow innermost) in the element dtype, exactly like
    aten's sequential CPU scatter, so results are bit-exact (fp64 included)
    without atomics.
    """
    pid_dhw = tl.program_id(0)
    pid_nc = tl.program_id(1)

    pid_n = pid_nc % N
    pid_c = pid_nc // N

    offs = pid_dhw * BLOCK_DHW + tl.arange(0, BLOCK_DHW)
    mask = offs < D_in * H_in * W_in

    id_ = offs // (H_in * W_in)
    rem = offs % (H_in * W_in)
    ih = rem // W_in
    iw = rem % W_in

    # Per-axis candidate output coordinates, ascending: reflection, center,
    # far reflection; the validity masks follow aten's CPU kernel conditions.
    od_top = pad_d0 - id_
    dtop_valid = (id_ > 0) & (id_ <= pad_d0)
    od_ctr = pad_d0 + id_
    od_bot = 2 * D_in - 2 - id_ + pad_d0
    dbot_valid = (id_ >= D_in - 1 - pad_d1) & (id_ < D_in - 1)

    oh_top = pad_h0 - ih
    htop_valid = (ih > 0) & (ih <= pad_h0)
    oh_ctr = pad_h0 + ih
    oh_bot = 2 * H_in - 2 - ih + pad_h0
    hbot_valid = (ih >= H_in - 1 - pad_h1) & (ih < H_in - 1)

    ow_top = pad_w0 - iw
    wtop_valid = (iw > 0) & (iw <= pad_w0)
    ow_ctr = pad_w0 + iw
    ow_bot = 2 * W_in - 2 - iw + pad_w0
    wbot_valid = (iw >= W_in - 1 - pad_w1) & (iw < W_in - 1)

    grad_base = pid_n * stride_grad_n + pid_c * stride_grad_c
    acc = tl.zeros((BLOCK_DHW,), dtype=grad_ptr.dtype.element_ty)

    for ds in tl.static_range(3):
        if ds == 0:
            od = od_top
            vd = dtop_valid
        elif ds == 1:
            od = od_ctr
            vd = mask
        else:
            od = od_bot
            vd = dbot_valid
        for hs in tl.static_range(3):
            if hs == 0:
                oh = oh_top
                vh = htop_valid
            elif hs == 1:
                oh = oh_ctr
                vh = mask
            else:
                oh = oh_bot
                vh = hbot_valid
            for ws in tl.static_range(3):
                if ws == 0:
                    ow = ow_top
                    vw = wtop_valid
                elif ws == 1:
                    ow = ow_ctr
                    vw = mask
                else:
                    ow = ow_bot
                    vw = wbot_valid
                v = mask & vd & vh & vw
                grad_off = od * stride_grad_d + oh * stride_grad_h + ow * stride_grad_w
                acc += tl.load(grad_ptr + grad_base + grad_off, mask=v, other=0.0)

    out_base = pid_n * stride_out_n + pid_c * stride_out_c
    out_off = id_ * stride_out_d + ih * stride_out_h + iw * stride_out_w
    tl.store(out_ptr + out_base + out_off, acc.to(out_ptr.dtype.element_ty), mask=mask)


def reflection_pad3d_backward(grad_output, self, padding):
    """Compute gradient of reflection_pad3d forward pass (aten::reflection_pad3d_backward).

    Padding order is ATEN's ``(pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1)``
    (last dim first). Upstream FlagGems took the diagonals depth-first
    ``(pad_d0, pad_d1, pad_h0, pad_h1, pad_w0, pad_w1)``; this wrapper accepts
    the aten order and reorders internally for the kernel.

    Other deviations from upstream, both to match aten: 4-D (C, D, H, W)
    inputs are accepted via unsqueeze (aten allows them; upstream required
    5-D), and the upstream output-space atomic kernel (float32 atomics, no
    tail mask -- out-of-bounds atomic writes on the last partial block) is
    replaced by a deterministic input-space gather accumulating in the
    element dtype.

    Args:
        grad_output: Gradient of the loss w.r.t. the output of reflection_pad3d.
                    Shape: (N, C, D + pad_d0 + pad_d1, H + pad_h0 + pad_h1, W + pad_w0 + pad_w1)
        self: Original input tensor before padding. Shape: (N, C, D, H, W)
        padding: Tuple of 6 ints in aten order (pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1)

    Returns:
        Gradient with respect to self. Shape: (N, C, D, H, W)
    """
    logger.debug("GEMS REFLECTION_PAD3D_BACKWARD")

    if isinstance(padding, int):
        pad_w0 = pad_w1 = pad_h0 = pad_h1 = pad_d0 = pad_d1 = padding
    else:
        pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1 = padding

    # Validate input (4-D is aten's unbatched form; handle it via unsqueeze)
    is_4d = self.dim() == 4
    if is_4d:
        self = self.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)
    if self.dim() != 5:
        raise ValueError("input must be a 5D tensor")

    N, C, D_in, H_in, W_in = self.shape
    D_out, H_out, W_out = (
        D_in + pad_d0 + pad_d1,
        H_in + pad_h0 + pad_h1,
        W_in + pad_w0 + pad_w1,
    )

    # Validate shapes
    expected_grad_shape = (N, C, D_out, H_out, W_out)
    if tuple(grad_output.shape) != expected_grad_shape:
        raise ValueError(
            f"grad_output has shape {tuple(grad_output.shape)}, expected {expected_grad_shape}"
        )

    # Handle empty padding case - just copy
    if (
        pad_d0 == 0
        and pad_d1 == 0
        and pad_h0 == 0
        and pad_h1 == 0
        and pad_w0 == 0
        and pad_w1 == 0
    ):
        return grad_output.clone()

    grad_output = grad_output.contiguous()

    grad_input = torch.empty_like(self)
    if grad_input.numel() == 0:
        return grad_input.squeeze(0) if is_4d else grad_input

    # Launch kernel with input-space parallelization
    BLOCK_DHW = 256
    grid = (
        triton.cdiv(D_in * H_in * W_in, BLOCK_DHW),
        N * C,
    )

    reflection_pad3d_backward_kernel[grid](
        grad_output,
        grad_input,
        N,
        C,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        pad_d0,
        pad_d1,
        pad_h0,
        pad_h1,
        pad_w0,
        pad_w1,
        *grad_output.stride(),
        *grad_input.stride(),
        BLOCK_DHW=BLOCK_DHW,
    )

    return grad_input.squeeze(0) if is_4d else grad_input
