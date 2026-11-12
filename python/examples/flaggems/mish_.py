import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def mish_kernel_(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x)), in fp32.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    y = xf * libdevice.tanh(tl.log(1.0 + tl.exp(xf)))

    tl.store(x_ptr + offsets, y.to(x.dtype), mask=mask)


def mish_(*args, **kwargs):
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
        raise ValueError("mish_: expected a Tensor as the first argument")

    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return torch.ops.aten.mish_(x)

    x_c = x.contiguous()
    n_elements = x_c.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    mish_kernel_[grid](x_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if x_c.data_ptr() != x.data_ptr():
        x.copy_(x_c)
    return x
