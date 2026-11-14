import torch
import triton
import triton.language as tl


@triton.jit
def hardshrink_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    lambd,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # hardshrink(x) = x if |x| > lambd else 0. The zero is built as x * 0 so
    # every branch stays in the input dtype (bfloat16 included, issue #114).
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    zeros = x * 0
    keep = (x > lambd) | (x < -lambd)
    y = tl.where(keep, x, zeros)

    tl.store(out_ptr + offsets, y, mask=mask)


def _hardshrink_impl(x: torch.Tensor, lambd, out=None):
    if not x.dtype.is_floating_point:
        raise NotImplementedError("hardshrink_cpu not implemented for integer inputs")
    x_c = x.contiguous()
    if out is None:
        out = torch.empty_like(x_c)
    out_c = out if out.is_contiguous() else out.contiguous()

    n_elements = x_c.numel()
    if out.numel() != n_elements:
        raise ValueError(
            "Output tensor must have the same number of elements as the broadcast input"
        )
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    hardshrink_kernel[grid](x_c, out_c, n_elements, lambd, BLOCK_SIZE=BLOCK_SIZE)

    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def hardshrink(x: torch.Tensor, lambd: float = 0.5):
    if not x.dtype.is_floating_point:
        # torch: '"hardshrink_cpu" not implemented for integer inputs'.
        raise NotImplementedError('"hardshrink_cpu" not implemented for integer inputs')
    return _hardshrink_impl(x, lambd)


def hardshrink_out(x: torch.Tensor, lambd: float = 0.5, *, out: torch.Tensor = None):
    return _hardshrink_impl(x, lambd, out=out)
