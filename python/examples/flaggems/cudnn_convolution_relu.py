import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _cudnn_convolution_relu_kernel(
    conv_ptr, bias_ptr, out_ptr, n_elements, spatial, channels, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    conv = tl.load(conv_ptr + offsets, mask=mask, other=0.0)
    channel = (offsets // spatial) % channels
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0)
    res = conv + bias
    tl.store(out_ptr + offsets, tl.where(res > 0, res, 0.0), mask=mask)


def _check_dtype(input):
    # cuDNN engines for the fused conv+bias+relu exist only for the three
    # float types (measured on ATen: fp64/int8 fail with "unable to find an
    # engine"), so anything else is rejected up front -- mirroring the
    # upstream FlagGems check.
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), (
        f"cudnn_convolution_relu only supports float16/bfloat16/float32, got {input.dtype}"
    )


def cudnn_convolution_relu(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
):
    """Fused convolution + bias + ReLU, cuDNN style.

    The convolution arithmetic is delegated to aten
    (torch.nn.functional.conv1d/conv2d/conv3d); only the bias-add and ReLU
    epilogue runs as the Triton kernel below. The Triton-path test therefore
    spies on the epilogue, not on the whole operator.

    Args:
        input: (N, C_in, *spatial_dims) with one to three spatial dims.
        weight: (C_out, C_in / groups, *kernel_dims).
        bias: optional one value per output channel.
        stride/padding/dilation/groups: forwarded to the aten convolution.

    Returns:
        relu(conv(input, weight) + bias), shape (N, C_out, *out_spatial).
    """
    _check_dtype(input)
    ndim = input.dim() - 2
    if ndim not in (1, 2, 3):
        raise ValueError(
            f"cudnn_convolution_relu only supports 1D, 2D, and 3D convolutions, "
            f"got input with {ndim} spatial dimensions"
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
    if bias is None:
        bias_c = torch.zeros(out_channels, dtype=conv.dtype, device=conv.device)
    else:
        bias_c = bias.reshape(-1).to(conv.dtype).contiguous()
        if bias_c.numel() != out_channels:
            raise RuntimeError(
                f"cudnn_convolution_relu: bias has {bias_c.numel()} values for "
                f"{out_channels} output channels"
            )
    out = torch.empty_like(conv)
    n = out.numel()
    if n:
        spatial = n // (conv.size(0) * out_channels)
        _cudnn_convolution_relu_kernel[(triton.cdiv(n, 1024),)](
            conv, bias_c, out, n, spatial, out_channels, BLOCK=1024
        )
    return out
