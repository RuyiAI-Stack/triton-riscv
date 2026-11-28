import torch
import triton
import triton.language as tl


@triton.jit
def _prelu_backward_kernel(
    g_ptr,
    x_ptr,
    w_ptr,
    gi_ptr,
    contrib_ptr,
    n_elements,
    spatial,
    channels,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    channel = (offsets // spatial) % channels
    w = tl.load(w_ptr + channel, mask=mask, other=0.0)
    positive = x >= 0
    tl.store(gi_ptr + offsets, tl.where(positive, g, g * w), mask=mask)
    # Per-element contribution to the weight gradient; the channel reduction
    # is summed on the host so the kernel stays a plain elementwise pass.
    tl.store(contrib_ptr + offsets, tl.where(positive, 0.0, g * x), mask=mask)


def _prelu_kernel_backward(*args, **kwargs):
    """grad_input and grad_weight for PReLU (aten::prelu_backward shape).

    grad_input = x >= 0 ? grad_output : grad_output * weight
    grad_weight[c] = sum over batch/spatial of grad_output * x where x < 0.
    """
    if len(args) >= 3:
        grad_output, x, weight = args[0], args[1], args[2]
    else:
        grad_output = kwargs.get("grad_output")
        x = kwargs.get("self", kwargs.get("x"))
        weight = kwargs.get("weight")
    if grad_output is None or x is None or weight is None:
        raise ValueError(
            "_prelu_kernel_backward expects (grad_output, self, weight) as arguments."
        )
    x_c = x.contiguous()
    g_c = grad_output.to(x.dtype).expand(x.shape).contiguous()
    w = weight.to(x.dtype).reshape(-1).contiguous()
    if x.dim() <= 1:
        channels = 1
        spatial = x.numel()
    else:
        channels = x.size(1)
        spatial = x.numel() // (x.size(0) * channels) if x.numel() and channels else 0
    if w.numel() == 1:
        w = w.expand(channels).contiguous()
    elif w.numel() != channels:
        raise ValueError(
            f"_prelu_kernel_backward: weight has {w.numel()} values for "
            f"{channels} channels"
        )
    grad_input = torch.empty_like(x_c)
    contrib = torch.empty_like(x_c)
    n = x_c.numel()
    if n:
        _prelu_backward_kernel[(triton.cdiv(n, 1024),)](
            g_c, x_c, w, grad_input, contrib, n, spatial, channels, BLOCK=1024
        )
    if x.dim() > 1:
        reduce_dims = [0] + list(range(2, x.dim()))
        grad_weight = contrib.sum(dim=reduce_dims)
    else:
        grad_weight = contrib.reshape(1)
    return grad_input, grad_weight.reshape(weight.shape)
