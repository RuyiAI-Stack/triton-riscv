import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def mish_kernel(
    x_ptr,
    out_ptr,
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

    tl.store(out_ptr + offsets, y.to(x.dtype), mask=mask)


def mish(x: torch.Tensor):
    if not x.dtype.is_floating_point:
        # torch: '"mish_cpu" not implemented for integer inputs'.
        raise NotImplementedError('"mish_cpu" not implemented for integer inputs')
    if x.dtype == torch.float64:
        return torch.nn.functional.mish(x)

    x_c = x.contiguous()
    out = torch.empty_like(x_c)

    n_elements = x_c.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    mish_kernel[grid](x_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out
