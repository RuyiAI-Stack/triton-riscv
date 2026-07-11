import torch
import triton
import triton.language as tl


def pool3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    numerator = in_size + 2 * padding - effective_kernel_size
    if ceil_mode:
        output_size = (numerator + stride - 1) // stride + 1
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
        return output_size
    return numerator // stride + 1


@triton.jit
def max_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
):
    output_offset = tl.program_id(0)
    w_out = output_offset % out_w
    remaining = output_offset // out_w
    h_out = remaining % out_h
    remaining //= out_h
    d_out = remaining % out_d
    nc_idx = remaining // out_d
    input_volume = in_d * in_h * in_w
    input_base = nc_idx * input_volume

    min_value: tl.constexpr = float("-inf")
    max_value = tl.full((), min_value, dtype=tl.float32)
    max_index = tl.full((), -1, dtype=tl.int32)
    for kd in tl.static_range(0, kernel_d):
        for kh in tl.static_range(0, kernel_h):
            for kw in tl.static_range(0, kernel_w):
                d_in = d_out * stride_d - padding_d + kd * dilation_d
                h_in = h_out * stride_h - padding_h + kh * dilation_h
                w_in = w_out * stride_w - padding_w + kw * dilation_w
                valid = (
                    (d_in >= 0)
                    & (d_in < in_d)
                    & (h_in >= 0)
                    & (h_in < in_h)
                    & (w_in >= 0)
                    & (w_in < in_w)
                )
                safe_d = tl.maximum(0, tl.minimum(d_in, in_d - 1))
                safe_h = tl.maximum(0, tl.minimum(h_in, in_h - 1))
                safe_w = tl.maximum(0, tl.minimum(w_in, in_w - 1))
                input_index = safe_d * in_h * in_w + safe_h * in_w + safe_w
                value = tl.load(input_ptr + input_base + input_index).to(
                    tl.float32
                )
                value = tl.where(valid, value, min_value)
                update = valid & ((max_index < 0) | (value > max_value))
                max_value = tl.where(update, value, max_value)
                flat_index = d_in * in_h * in_w + h_in * in_w + w_in
                max_index = tl.where(update, flat_index, max_index)

    tl.store(output_ptr + output_offset, max_value)
    tl.store(indices_ptr + output_offset, max_index)


@triton.jit
def max_pool3d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
):
    input_offset = tl.program_id(0)
    input_volume = in_d * in_h * in_w
    input_index = input_offset % input_volume
    nc_idx = input_offset // input_volume
    output_volume = out_d * out_h * out_w
    grad = tl.full((), 0.0, dtype=tl.float32)
    for output_index in range(0, output_volume):
        offset = nc_idx * output_volume + output_index
        selected_index = tl.load(indices_ptr + offset).to(tl.int32)
        output_grad = tl.load(grad_output_ptr + offset).to(tl.float32)
        grad += tl.where(selected_index == input_index, output_grad, 0.0)
    tl.store(grad_input_ptr + input_offset, grad)


def _parse_pool3d_params(kernel_size, stride, padding, dilation):
    def _parse_param(param, name, default=None):
        if param is None:
            return default
        if isinstance(param, int):
            return param, param, param
        if isinstance(param, (list, tuple)) and len(param) == 3:
            return tuple(param)
        raise ValueError(f"Invalid {name}: {param}")

    kd, kh, kw = _parse_param(kernel_size, "kernel_size")
    sd, sh, sw = _parse_param(stride, "stride", default=(kd, kh, kw))
    pd, ph, pw = _parse_param(padding, "padding", default=(0, 0, 0))
    dd, dh, dw = _parse_param(dilation, "dilation", default=(1, 1, 1))
    if sd <= 0 or sh <= 0 or sw <= 0:
        raise ValueError("stride must be positive")
    if pd < 0 or ph < 0 or pw < 0:
        raise ValueError("padding must be non-negative")
    if dd <= 0 or dh <= 0 or dw <= 0:
        raise ValueError("dilation must be positive")
    return kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw


def max_pool3d_with_indices(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    input = input.contiguous()
    params = _parse_pool3d_params(kernel_size, stride, padding, dilation)
    kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw = params
    in_n, in_c, in_d, in_h, in_w = input.shape
    out_d = pool3d_output_size(in_d, kd, sd, pd, dd, ceil_mode)
    out_h = pool3d_output_size(in_h, kh, sh, ph, dh, ceil_mode)
    out_w = pool3d_output_size(in_w, kw, sw, pw, dw, ceil_mode)
    output = torch.empty(
        (in_n, in_c, out_d, out_h, out_w),
        device=input.device,
        dtype=input.dtype,
    )
    indices = torch.empty(
        output.shape, device=input.device, dtype=torch.int64
    )
    if output.numel() == 0:
        return output, indices
    max_pool3d_forward_kernel[(output.numel(),)](
        input,
        output,
        indices,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        kernel_d=kd,
        kernel_h=kh,
        kernel_w=kw,
        stride_d=sd,
        stride_h=sh,
        stride_w=sw,
        padding_d=pd,
        padding_h=ph,
        padding_w=pw,
        dilation_d=dd,
        dilation_h=dh,
        dilation_w=dw,
    )
    return output, indices


def max_pool3d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    indices: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    grad_output = grad_output.contiguous()
    indices = indices.contiguous()
    in_n, in_c, in_d, in_h, in_w = input.shape
    out_d, out_h, out_w = grad_output.shape[2:]
    grad_input = torch.empty_like(input, memory_format=torch.contiguous_format)
    if grad_input.numel() == 0:
        return grad_input
    max_pool3d_backward_kernel[(grad_input.numel(),)](
        grad_output,
        indices,
        grad_input,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
    )
    return grad_input
