import torch
import triton
import triton.language as tl


@triton.jit
def col2im_kernel(
    input_ptr,
    output_ptr,
    in_stride_n,
    in_stride_ck,
    in_stride_l,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    batch_size,
    channels,
    out_h,
    out_w,
    L_h,
    L_w,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    n_idx = pid_nc // channels
    c_idx = pid_nc % channels

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    sum_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    input_base_ptr = input_ptr + n_idx * in_stride_n

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_num = h_out_offsets[:, None] + padding_h - kh * dilation_h
            w_num = w_out_offsets[None, :] + padding_w - kw * dilation_w

            h_valid = (h_num % stride_h) == 0
            w_valid = (w_num % stride_w) == 0

            l_h = h_num // stride_h
            l_w = w_num // stride_w

            l_h_valid = (l_h >= 0) & (l_h < L_h)
            l_w_valid = (l_w >= 0) & (l_w < L_w)

            valid_mask = h_valid & w_valid & l_h_valid & l_w_valid

            c_k = c_idx * kernel_h * kernel_w + kh * kernel_w + kw
            l_idx = l_h * L_w + l_w

            input_offset = c_k * in_stride_ck + l_idx * in_stride_l

            input_val = tl.load(
                input_base_ptr + input_offset, mask=valid_mask, other=0.0
            )

            sum_acc += input_val

    out_base_ptr = output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
    out_offset = (
        h_out_offsets[:, None] * out_stride_h
        + w_out_offsets[None, :] * out_stride_w
    )

    out_mask = (h_out_offsets[:, None] < out_h) & (
        w_out_offsets[None, :] < out_w
    )
    tl.store(
        out_base_ptr + out_offset,
        sum_acc.to(output_ptr.type.element_ty),
        mask=out_mask,
    )


def parse_col2im_params(output_size, kernel_size, dilation, padding, stride):
    def to_pair(val, name):
        if isinstance(val, int):
            return val, val
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return tuple(val)
        raise ValueError(f"Invalid {name}: {val}")

    out_h, out_w = to_pair(output_size, "output_size")
    kernel_h, kernel_w = to_pair(kernel_size, "kernel_size")
    dilation_h, dilation_w = to_pair(dilation, "dilation")
    padding_h, padding_w = to_pair(padding, "padding")
    stride_h, stride_w = to_pair(stride, "stride")

    if stride_h <= 0 or stride_w <= 0:
        raise ValueError(
            f"stride must be positive, got ({stride_h}, {stride_w})"
        )
    if padding_h < 0 or padding_w < 0:
        raise ValueError(
            f"padding must be non-negative, got ({padding_h}, {padding_w})"
        )
    if dilation_h <= 0 or dilation_w <= 0:
        raise ValueError(
            f"dilation must be positive, got ({dilation_h}, {dilation_w})"
        )

    return (
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        dilation_h,
        dilation_w,
        padding_h,
        padding_w,
        stride_h,
        stride_w,
    )


def col2im(input, output_size, kernel_size, dilation, padding, stride):
    (
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        dilation_h,
        dilation_w,
        padding_h,
        padding_w,
        stride_h,
        stride_w,
    ) = parse_col2im_params(
        output_size, kernel_size, dilation, padding, stride
    )

    if input.dim() != 3:
        raise ValueError(f"Expected 3D input, got {input.dim()}D")

    batch_size, ck, L = input.shape

    L_h = (
        out_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    L_w = (
        out_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    expected_L = L_h * L_w

    if L != expected_L:
        raise ValueError(
            f"Input size mismatch: expected L={expected_L} (L_h={L_h}, L_w={L_w}), got L={L}"
        )

    kernel_size_total = kernel_h * kernel_w
    if ck % kernel_size_total != 0:
        raise ValueError(
            f"Input dimension 1 ({ck}) must be divisible by "
            f"kernel_size ({kernel_size_total})"
        )
    channels = ck // kernel_size_total

    input = input.contiguous()

    output = torch.empty(
        (batch_size, channels, out_h, out_w),
        device=input.device,
        dtype=input.dtype,
    )

    if output.numel() == 0:
        return output

    BLOCK_H = 32
    BLOCK_W = 32
    grid = (
        batch_size * channels,
        triton.cdiv(out_h, BLOCK_H) * triton.cdiv(out_w, BLOCK_W),
    )

    col2im_kernel[grid](
        input,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        batch_size,
        channels,
        out_h,
        out_w,
        L_h,
        L_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )

    return output
