import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["pad_left", "pad_top", "pad_front"])
def _replication_pad3d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    B,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    pad_left,
    pad_top,
    pad_front,
    out_stride_b,
    in_stride_b,
    BLOCK_W: tl.constexpr,
):
    """Input-space gather for the replication_pad3d backward.

    Each program owns one input row (b, d, h) and a BLOCK_W tile of input
    columns. It sweeps grad_output in ascending (od, oh, ow) order and adds,
    per input lane, every output voxel whose replicated source is (d, h, w).
    The adds therefore follow aten's exact CPU accumulation order in the
    element dtype -- bit-exact for fp64, no atomics. Upstream scattered into a
    float32 buffer with atomic_add (nondeterministic order, and fp64 inputs
    were rejected outright).
    """
    pid_row = tl.program_id(0)
    pid_w = tl.program_id(1)

    dhw = D_in * H_in
    d = (pid_row % dhw) // H_in
    b = pid_row // dhw
    h = pid_row % H_in

    iw = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = iw < W_in

    grad_base = grad_output_ptr + b * out_stride_b

    # Scalar (od, oh, ow) sweep: each scalar grad value is broadcast onto the
    # iw lanes whose replicated source it hits. Per input lane the adds then
    # run in aten's exact ascending (od, oh, ow) order in the element dtype.
    acc = tl.zeros((BLOCK_W,), dtype=grad_output_ptr.dtype.element_ty)
    for od in range(0, D_out):
        target_d = tl.minimum(tl.maximum(od - pad_front, 0), D_in - 1)
        d_hit = target_d == d
        for oh in range(0, H_out):
            target_h = tl.minimum(tl.maximum(oh - pad_top, 0), H_in - 1)
            h_hit = target_h == h
            for ow in range(0, W_out):
                target_iw = tl.minimum(tl.maximum(ow - pad_left, 0), W_in - 1)
                hit = mask & d_hit & h_hit & (target_iw == iw)
                g = tl.load(grad_base + (od * H_out + oh) * W_out + ow)
                acc += tl.where(hit, g, 0.0)

    grad_in_base = grad_input_ptr + b * in_stride_b + (d * H_in + h) * W_in
    tl.store(
        grad_in_base + iw,
        acc.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _copy_rows_kernel(in_ptr, out_ptr, B, N, BLOCK_W: tl.constexpr):
    pid_b = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (offs_w < N) & (pid_b < B)

    base = pid_b * N
    vals = tl.load(in_ptr + base + offs_w, mask=mask, other=0)
    tl.store(out_ptr + base + offs_w, vals, mask=mask)


def _launch(grad_output: torch.Tensor, self: torch.Tensor, padding):
    if not isinstance(padding, (list, tuple)) or len(padding) != 6:
        raise ValueError(
            "padding must be a sequence of length 6: "
            "(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)"
        )
    pad_left = int(padding[0])
    pad_right = int(padding[1])
    pad_top = int(padding[2])
    pad_bottom = int(padding[3])
    pad_front = int(padding[4])
    pad_back = int(padding[5])
    if self.dim() < 3:
        raise ValueError("self must have at least 3 dimensions (D, H, W)")
    if grad_output.device != self.device or grad_output.dtype != self.dtype:
        raise ValueError("grad_output and self must have the same device and dtype")
    if self.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(
            "replication_pad3d_backward supports float16, bfloat16, float32 and float64"
        )

    grad_output_c = grad_output.contiguous()
    x = self.contiguous()

    D_in = int(x.shape[-3])
    H_in = int(x.shape[-2])
    W_in = int(x.shape[-1])
    if D_in <= 0 or H_in <= 0 or W_in <= 0:
        raise ValueError("input volumetric dims (D, H, W) must all be > 0")

    D_out = D_in + pad_front + pad_back
    H_out = H_in + pad_top + pad_bottom
    W_out = W_in + pad_left + pad_right
    if D_out <= 0 or H_out <= 0 or W_out <= 0:
        raise ValueError("padding results in a non-positive output dimension")
    expected = (D_out, H_out, W_out)
    actual = (
        int(grad_output_c.shape[-3]),
        int(grad_output_c.shape[-2]),
        int(grad_output_c.shape[-1]),
    )
    if actual != expected:
        raise ValueError(
            f"grad_output spatial shape {actual} does not match expected {expected}."
        )
    if tuple(grad_output_c.shape[:-3]) != tuple(x.shape[:-3]):
        raise ValueError("grad_output and self must have matching leading dimensions")

    leading_shape = x.shape[:-3]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    grad_input = torch.empty_like(x)

    all_zero = (pad_left | pad_right | pad_top | pad_bottom | pad_front | pad_back) == 0
    if all_zero:
        # Degenerate case: backward is a plain copy.
        n_per_row = D_in * H_in * W_in
        grid = (B, triton.cdiv(n_per_row, 256))
        _copy_rows_kernel[grid](grad_output_c, grad_input, B, n_per_row, BLOCK_W=256)
        return grad_input

    if grad_input.numel() > 0:
        BLOCK_W = 256
        grid = (B * D_in * H_in, triton.cdiv(W_in, BLOCK_W))
        _replication_pad3d_backward_kernel[grid](
            grad_output_c,
            grad_input,
            B,
            D_in,
            H_in,
            W_in,
            D_out,
            H_out,
            W_out,
            pad_left,
            pad_top,
            pad_front,
            D_out * H_out * W_out,
            D_in * H_in * W_in,
            BLOCK_W=BLOCK_W,
        )
    return grad_input


def replication_pad3d_backward(
    grad_output: torch.Tensor, self: torch.Tensor, padding
) -> torch.Tensor:
    """aten::replication_pad3d_backward; padding order (l, r, t, b, f, b) like aten.

    For a volumetric input with shape ``(*, D_in, H_in, W_in)`` the forward
    maps each output position ``(z, y, x)`` to the input position
    ``(clamp(z - pad_front, 0, D_in - 1), clamp(y - pad_top, 0, H_in - 1),
    clamp(x - pad_left, 0, W_in - 1))``; the backward accumulates every
    ``grad_output[z, y, x]`` into its clamped input cell, in ascending output
    order and in the element dtype (aten's CPU semantics).

    Deviations from upstream to match the aten oracle: float64 inputs are
    supported (aten accepts them; upstream rejected), and the float32
    atomic-add scatter is replaced by the deterministic gather above. The
    dtype error message is kept verbatim from upstream.
    """
    logger.debug("GEMS REPLICATION_PAD3D_BACKWARD")
    return _launch(grad_output, self, padding)
