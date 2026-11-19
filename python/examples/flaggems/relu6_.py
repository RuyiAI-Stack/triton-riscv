import torch
import triton
import triton.language as tl


@triton.jit
def relu6_kernel_(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # relu6(x) = clamp(x, 0, 6) in the input dtype; NaN restored because
    # PyTorch's clamp propagates it while tl.maximum drops it.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.minimum(tl.maximum(x, 0), 6)
    y = tl.where(x != x, x, y)

    tl.store(x_ptr + offsets, y, mask=mask)


def relu6_(*args, **kwargs):
    x = None
    if len(args) > 0 and isinstance(args[0], torch.Tensor):
        x = args[0]
    else:
        for key in ("input", "self", "x"):
            val = kwargs.get(key, None)
            if isinstance(val, torch.Tensor):
                x = val
                break
    if x is None:
        raise ValueError("relu6_: expected a Tensor as the first argument")

    x_c = x.contiguous()
    n_elements = x_c.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    relu6_kernel_[grid](x_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if x_c.data_ptr() != x.data_ptr():
        x.copy_(x_c)
    return x
