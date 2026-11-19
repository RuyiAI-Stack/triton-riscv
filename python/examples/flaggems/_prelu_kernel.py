import torch
import triton
import triton.language as tl


@triton.jit
def _prelu_forward_kernel(
    x_ptr, w_ptr, out_ptr, n_elements, spatial, channels, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    channel = (offsets // spatial) % channels
    w = tl.load(w_ptr + channel, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, tl.where(x >= 0, x, x * w), mask=mask)


def _prelu_kernel(A, B):
    """PReLU: x >= 0 -> x, x < 0 -> x * weight (aten::_prelu_kernel).

    The weight is either a single value or one value per channel (dim 1);
    for 1-D inputs the channel axis degenerates to the single element axis.
    """
    x = A.contiguous()
    if not isinstance(B, torch.Tensor):
        B = torch.tensor(B, dtype=x.dtype)
    weight = B.to(x.dtype).reshape(-1).contiguous()
    if x.dim() <= 1:
        channels = 1
        spatial = x.numel()
    else:
        channels = x.size(1)
        spatial = x.numel() // (x.size(0) * channels) if x.numel() and channels else 0
    if weight.numel() == 1:
        weight = weight.expand(channels).contiguous()
    elif weight.numel() != channels:
        raise ValueError(
            f"_prelu_kernel: weight has {weight.numel()} values for {channels} channels"
        )
    out = torch.empty_like(x)
    n = out.numel()
    if n:
        _prelu_forward_kernel[(triton.cdiv(n, 1024),)](
            x, weight, out, n, spatial, channels, BLOCK=1024
        )
    return out
