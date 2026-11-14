import torch
import triton
import triton.language as tl


@triton.jit
def hardtanh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # hardtanh(x) = clamp(x, min_val, max_val) in the input dtype, so integer
    # inputs stay integers. PyTorch's clamp propagates NaN while
    # tl.maximum/tl.minimum drop it, so NaN is restored.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.minimum(tl.maximum(x, min_val), max_val)
    y = tl.where(x != x, x, y)

    tl.store(out_ptr + offsets, y, mask=mask)


def _hardtanh_impl(x: torch.Tensor, min_val, max_val, out=None):
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
    hardtanh_kernel[grid](
        x_c, out_c, n_elements, min_val, max_val, BLOCK_SIZE=BLOCK_SIZE
    )

    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def hardtanh(x: torch.Tensor, min_val=-1.0, max_val=1.0):
    return _hardtanh_impl(x, min_val, max_val)


def hardtanh_out(
    x: torch.Tensor, min_val=-1.0, max_val=1.0, *, out: torch.Tensor = None
):
    return _hardtanh_impl(x, min_val, max_val, out=out)
