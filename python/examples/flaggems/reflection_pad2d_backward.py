import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def reflection_pad2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    # Shapes
    in_h,
    in_w,
    out_h,
    out_w,
    # Strides for grad_output
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    # Strides for grad_input
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    # Number of channels
    num_c,
    # Padding
    pad_left,
    pad_right,
    pad_top,
    pad_bottom,
    # Block sizes
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program ID: (n * c, h_blocks * w_blocks)
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_w_blocks = tl.cdiv(in_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // num_c
    c_idx = pid_nc % num_c

    # Input position offsets within this block
    h_in_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_in_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Initialize gradient accumulator in the element dtype: aten's CPU
    # backward sums the (up to 9) contributions in scalar_t, in ascending
    # output (row-major) order, so the adds below follow the order
    # tl, tc, tr, ml, mc, mr, bl, bc, br to be bit-exact (float64 included).
    grad_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=grad_output_ptr.dtype.element_ty)

    # Base pointer for grad_output for this (n, c)
    grad_out_base = grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c

    # For each input position, we need to find all output positions that map to it
    # In forward pass:
    #   output[out_h, out_w] = input[reflect(out_h - pad_top), reflect(out_w - pad_left)]
    # In backward:
    #   grad_input[h_in, w_in] = sum of grad_output[out_h, out_w]
    #     for all (out_h, out_w) that map to (h_in, w_in)

    # The output positions that map to input position h_in are:
    # 1. Direct: out_h = h_in + pad_top
    # 2. Top reflection: out_h = pad_top - h_in (if h_in > 0 and h_in <= pad_top)
    # 3. Bottom reflection: out_h = 2*in_h - 2 - h_in + pad_top (if h_in >= in_h - pad_bottom - 1 and h_in < in_h - 1)

    # Process height positions
    h_in_2d = h_in_offsets[:, None]
    w_in_2d = w_in_offsets[None, :]

    # Create 2D masks for valid input positions
    h_valid = h_in_offsets < in_h
    w_valid = w_in_offsets < in_w
    valid_mask = h_valid[:, None] & w_valid[None, :]

    # For each input h_in, there are up to 3 output h positions:
    # h_out_center = h_in + pad_top (always valid)
    # h_out_top = pad_top - h_in (valid if h_in > 0 and h_in <= pad_top)
    # h_out_bottom = 2*in_h - 2 - h_in + pad_top (valid if h_in >= in_h - 1 - pad_bottom and h_in < in_h - 1)

    # Similarly for width

    # Center position (always contributes)
    h_out_center = h_in_2d + pad_top
    w_out_center = w_in_2d + pad_left

    # Top reflection contribution
    # Valid if h_in > 0 and h_in <= pad_top
    h_out_top = pad_top - h_in_2d
    top_h_valid = (h_in_2d > 0) & (h_in_2d <= pad_top)

    # Bottom reflection contribution
    # Valid if h_in >= in_h - 1 - pad_bottom and h_in < in_h - 1
    h_out_bottom = 2 * in_h - 2 - h_in_2d + pad_top
    bottom_h_valid = (h_in_2d >= in_h - 1 - pad_bottom) & (h_in_2d < in_h - 1)

    # Left reflection contribution (with center height)
    w_out_left = pad_left - w_in_2d
    left_w_valid = (w_in_2d > 0) & (w_in_2d <= pad_left)

    # Right reflection contribution (with center height)
    w_out_right = 2 * in_w - 2 - w_in_2d + pad_left
    right_w_valid = (w_in_2d >= in_w - 1 - pad_right) & (w_in_2d < in_w - 1)

    # top-left
    tl_mask = top_h_valid & left_w_valid & valid_mask
    tl_offset = h_out_top * out_stride_h + w_out_left * out_stride_w
    tl_val = tl.load(grad_out_base + tl_offset, mask=tl_mask, other=0.0)
    grad_acc += tl_val

    # top-center
    tc_mask = top_h_valid & valid_mask
    tc_offset = h_out_top * out_stride_h + w_out_center * out_stride_w
    tc_val = tl.load(grad_out_base + tc_offset, mask=tc_mask, other=0.0)
    grad_acc += tc_val

    # top-right
    tr_mask = top_h_valid & right_w_valid & valid_mask
    tr_offset = h_out_top * out_stride_h + w_out_right * out_stride_w
    tr_val = tl.load(grad_out_base + tr_offset, mask=tr_mask, other=0.0)
    grad_acc += tr_val

    # middle-left (center height, left reflection)
    ml_mask = left_w_valid & valid_mask
    ml_offset = h_out_center * out_stride_h + w_out_left * out_stride_w
    ml_val = tl.load(grad_out_base + ml_offset, mask=ml_mask, other=0.0)
    grad_acc += ml_val

    # middle-center
    mc_offset = h_out_center * out_stride_h + w_out_center * out_stride_w
    mc_val = tl.load(grad_out_base + mc_offset, mask=valid_mask, other=0.0)
    grad_acc += mc_val

    # middle-right
    mr_mask = right_w_valid & valid_mask
    mr_offset = h_out_center * out_stride_h + w_out_right * out_stride_w
    mr_val = tl.load(grad_out_base + mr_offset, mask=mr_mask, other=0.0)
    grad_acc += mr_val

    # bottom-left
    bl_mask = bottom_h_valid & left_w_valid & valid_mask
    bl_offset = h_out_bottom * out_stride_h + w_out_left * out_stride_w
    bl_val = tl.load(grad_out_base + bl_offset, mask=bl_mask, other=0.0)
    grad_acc += bl_val

    # bottom-center
    bc_mask = bottom_h_valid & valid_mask
    bc_offset = h_out_bottom * out_stride_h + w_out_center * out_stride_w
    bc_val = tl.load(grad_out_base + bc_offset, mask=bc_mask, other=0.0)
    grad_acc += bc_val

    # bottom-right
    br_mask = bottom_h_valid & right_w_valid & valid_mask
    br_offset = h_out_bottom * out_stride_h + w_out_right * out_stride_w
    br_val = tl.load(grad_out_base + br_offset, mask=br_mask, other=0.0)
    grad_acc += br_val

    # Store result
    grad_in_base = grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    store_offset = (
        h_in_offsets[:, None] * in_stride_h + w_in_offsets[None, :] * in_stride_w
    )
    tl.store(
        grad_in_base + store_offset,
        grad_acc.to(grad_input_ptr.dtype.element_ty),
        mask=valid_mask,
    )


def reflection_pad2d_backward(grad_output, self, padding):
    """Backward pass for reflection_pad2d (aten::reflection_pad2d_backward).

    Padding order is aten's: ``[pad_left, pad_right, pad_top, pad_bottom]``.

    Args:
        grad_output: Gradient of loss w.r.t. padded output, shape (N, C, H_out, W_out)
        self: Original input tensor (used only for shape), shape (N, C, H, W)
        padding: [pad_left, pad_right, pad_top, pad_bottom]

    Returns:
        Gradient of loss w.r.t. input, shape (N, C, H, W)

    The 9 contributions per input cell are summed in ascending output order in
    the element dtype, matching aten's CPU kernel bit-for-bit (fp64 included).
    """
    logger.debug("GEMS REFLECTION_PAD2D_BACKWARD")

    grad_output = grad_output.contiguous()

    # Parse padding
    if len(padding) != 4:
        raise ValueError("padding must be a sequence of 4 elements")

    pad_left, pad_right, pad_top, pad_bottom = [int(p) for p in padding]

    # Get shapes
    if grad_output.dim() == 3:
        # (C, H, W) format
        num_c, out_h, out_w = grad_output.shape
        in_h = self.shape[-2]
        in_w = self.shape[-1]
        batch_size = 1
        grad_output = grad_output.unsqueeze(0)
        squeeze_output = True
    else:
        # (N, C, H, W) format
        batch_size, num_c, out_h, out_w = grad_output.shape
        in_h = self.shape[-2]
        in_w = self.shape[-1]
        squeeze_output = False

    # Create output tensor
    grad_input = torch.empty(
        (batch_size, num_c, in_h, in_w),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )

    if grad_input.numel() == 0:
        if squeeze_output:
            return grad_input.squeeze(0)
        return grad_input

    # Launch kernel
    BLOCK_H = 16
    BLOCK_W = 16
    grid = (
        batch_size * num_c,
        triton.cdiv(in_h, BLOCK_H) * triton.cdiv(in_w, BLOCK_W),
    )

    reflection_pad2d_backward_kernel[grid](
        grad_output,
        grad_input,
        in_h,
        in_w,
        out_h,
        out_w,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_input.stride(0),
        grad_input.stride(1),
        grad_input.stride(2),
        grad_input.stride(3),
        num_c,
        pad_left,
        pad_right,
        pad_top,
        pad_bottom,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )

    if squeeze_output:
        return grad_input.squeeze(0)
    return grad_input
