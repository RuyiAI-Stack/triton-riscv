import torch
import triton
import triton.language as tl


@triton.jit
def dequantize_kernel(
    x, scale, zero_point, output, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_val = tl.load(x + offsets, mask=mask, other=0).to(tl.float32)
    result = (x_val - zero_point) * scale
    tl.store(output + offsets, result, mask=mask)


@triton.jit
def dequantize_per_channel_kernel(
    x,
    scales,
    zero_points,
    output,
    inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    inner_block = tl.program_id(axis=0)
    channel_idx = tl.program_id(axis=1)
    outer_idx = tl.program_id(axis=2)
    offsets = inner_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < inner_size
    base = (outer_idx * tl.num_programs(axis=1) + channel_idx) * inner_size
    x_val = tl.load(x + base + offsets, mask=mask, other=0).to(tl.float32)
    scale = tl.load(scales + channel_idx)
    zero_point = tl.load(zero_points + channel_idx)
    result = (x_val - zero_point) * scale
    tl.store(output + base + offsets, result, mask=mask)


def _launch_dequantize_per_tensor(int_repr, scale, zero_point):
    output = torch.empty(int_repr.shape, dtype=torch.float32, device=int_repr.device)
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    dequantize_kernel[grid](int_repr, scale, zero_point, output, n_elements, BLOCK_SIZE)
    return output


def _launch_dequantize_per_channel(int_repr, scales, zero_points, axis):
    output = torch.empty(int_repr.shape, dtype=torch.float32, device=int_repr.device)
    BLOCK_SIZE = 1024
    inner_size = 1
    for dim in int_repr.shape[axis + 1 :]:
        inner_size *= int(dim)
    axis_size = int(int_repr.shape[axis])
    outer_size = int(int_repr.numel() // (axis_size * inner_size))
    grid = (triton.cdiv(inner_size, BLOCK_SIZE), axis_size, outer_size)
    dequantize_per_channel_kernel[grid](
        int_repr.reshape(-1),
        scales,
        zero_points,
        output.reshape(-1),
        inner_size,
        BLOCK_SIZE,
    )
    return output


def dequantize(a):
    """Dequantize a quantized tensor.

    Args:
        a: A quantized tensor (e.g., torch.qint8, torch.quint8)

    Returns:
        A float32 tensor with the dequantized values
    """
    if a.qscheme() in (
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ):
        int_repr = a.int_repr().to(a.device)
        scales = a.q_per_channel_scales().to(device=a.device, dtype=torch.float32)
        zero_points = a.q_per_channel_zero_points().to(
            device=a.device, dtype=torch.float32
        )
        axis = int(a.q_per_channel_axis())
        return _launch_dequantize_per_channel(int_repr, scales, zero_points, axis)

    int_repr = a.int_repr().to(a.device)
    scale = float(a.q_scale())
    zero_point = int(a.q_zero_point())
    return _launch_dequantize_per_tensor(int_repr, scale, zero_point)
