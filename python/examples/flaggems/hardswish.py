import torch
import triton
import triton.language as tl


@triton.jit
def hardswish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # hardswish(x) = x * relu6(x + 3) / 6, computed in fp32 so fp16 inputs
    # match PyTorch's accumulation precision. PyTorch's clamp propagates NaN
    # while tl.maximum drops it, so NaN is restored explicitly.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    inner = tl.minimum(tl.maximum(xf + 3.0, 0.0), 6.0)
    y = xf * inner * (1.0 / 6.0)
    y = tl.where(xf != xf, xf, y)

    tl.store(out_ptr + offsets, y.to(x.dtype), mask=mask)


def _hardswish_impl(x: torch.Tensor, out=None):
    if not x.dtype.is_floating_point:
        raise NotImplementedError("hardswish_cpu not implemented for integer inputs")
    if x.dtype == torch.float64:
        # The kernel computes in fp32; keep full fp64 precision via torch.
        result = torch.nn.functional.hardswish(x)
        if out is not None:
            out.copy_(result)
            return out
        return result

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
    hardswish_kernel[grid](x_c, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def hardswish(x: torch.Tensor):
    if not x.dtype.is_floating_point:
        # torch: '"hardswish_cpu" not implemented for integer inputs'.
        raise NotImplementedError('"hardswish_cpu" not implemented for integer inputs')
    return _hardswish_impl(x)


def hardswish_out(x: torch.Tensor, out: torch.Tensor):
    return _hardswish_impl(x, out=out)
