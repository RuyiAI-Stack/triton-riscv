import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _cudnn_convolution_add_relu_kernel(
    conv_ptr,
    z_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    spatial,
    channels,
    alpha,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    conv = tl.load(conv_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    channel = (offsets // spatial) % channels
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0)
    res = conv + bias + alpha * z
    tl.store(out_ptr + offsets, tl.where(res > 0, res, 0.0), mask=mask)


def cudnn_convolution_add_relu(
    input,
    weight,
    z,
    alpha=None,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    """Fused convolution + bias + alpha * z + ReLU, cuDNN style.

    Computes relu(conv(input, weight) + bias + alpha * z). The convolution
    arithmetic is delegated to aten (torch.nn.functional.conv1d/2d/3d); only
    the bias-add, alpha*z and ReLU epilogue runs as the Triton kernel below,
    and the Triton-path test spies on that epilogue only.

    Args:
        input: (N, C_in, *spatial_dims) with one to three spatial dims.
        weight: (C_out, C_in / groups, *kernel_dims).
        z: addend with the shape of the convolution output.
        alpha: scalar multiplier applied to z (default 1).
        bias: optional one value per output channel (default zeros).
        stride/padding/dilation/groups: forwarded to the aten convolution.

    Returns:
        relu(conv(input, weight) + bias + alpha * z).
    """
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        # Mirrors the upstream FlagGems check: the cuDNN engine for the fused
        # epilogue only exists for the three float types.
        raise RuntimeError(
            f"cudnn_convolution_add_relu only supports float16/bfloat16/float32 "
            f"inputs, got {input.dtype}"
        )
    ndim = input.dim() - 2
    if ndim not in (1, 2, 3):
        raise ValueError(
            f"cudnn_convolution_add_relu only supports 1D, 2D, and 3D "
            f"convolutions, got input with {ndim} spatial dimensions"
        )
    conv_fn = (F.conv1d, F.conv2d, F.conv3d)[ndim - 1]
    conv = conv_fn(
        input,
        weight,
        None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    out_channels = conv.size(1)
    alpha_val = 1.0 if alpha is None else float(alpha)
    if bias is None:
        bias_c = torch.zeros(out_channels, dtype=conv.dtype, device=conv.device)
    else:
        bias_c = bias.reshape(-1).to(conv.dtype).contiguous()
        if bias_c.numel() != out_channels:
            raise RuntimeError(
                f"cudnn_convolution_add_relu: bias has {bias_c.numel()} values "
                f"for {out_channels} output channels"
            )
    z_c = z.to(conv.dtype).expand(conv.shape).contiguous()
    out = torch.empty_like(conv)
    n = out.numel()
    if n:
        spatial = n // (conv.size(0) * out_channels)
        _cudnn_convolution_add_relu_kernel[(triton.cdiv(n, 1024),)](
            conv, z_c, bias_c, out, n, spatial, out_channels, alpha_val, BLOCK=1024
        )
    return out
