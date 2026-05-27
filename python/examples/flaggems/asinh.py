import torch
import triton
import triton.language as tl


@triton.jit
def asinh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    COMPUTE_FP32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    if COMPUTE_FP32:
        x32 = x.to(tl.float32)
        abs_x = tl.abs(x32)
        y32 = tl.log(abs_x + tl.sqrt(abs_x * abs_x + 1.0))
        y32 = tl.where(x32 < 0.0, -y32, y32)
        y = y32.to(x.dtype)
    else:
        abs_x = tl.abs(x)
        y = tl.log(abs_x + tl.sqrt(abs_x * abs_x + 1.0))
        y = tl.where(x < 0.0, -y, y)

    tl.store(out_ptr + offsets, y, mask=mask)


def asinh(x: torch.Tensor):
    if not x.dtype.is_floating_point:
        out = torch.empty_like(x, dtype=torch.float32)
        x = x.to(torch.float32)
    else:
        out = torch.empty_like(x)

    n_elements = x.numel()
    if n_elements == 0:
        return out

    x_c = x.contiguous()
    out_c = out if out.is_contiguous() else out.contiguous()

    use_fp32 = x.dtype in (torch.float16, torch.bfloat16)
    BLOCK_SIZE = 1024

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    asinh_kernel[grid](
        x_c, out_c, n_elements, COMPUTE_FP32=use_fp32, BLOCK_SIZE=BLOCK_SIZE
    )

    if not out.is_contiguous():
        out.copy_(out_c)

    return out


def asinh_out(x: torch.Tensor, out: torch.Tensor):
    if not x.dtype.is_floating_point:
        x = x.to(torch.float32)

    if out.dtype != x.dtype:
        out = out.to(x.dtype)

    n_elements = x.numel()
    if n_elements == 0:
        return out

    x_c = x.contiguous()
    out_c = out if out.is_contiguous() else out.contiguous()

    use_fp32 = x.dtype in (torch.float16, torch.bfloat16)
    BLOCK_SIZE = 1024

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    asinh_kernel[grid](
        x_c, out_c, n_elements, COMPUTE_FP32=use_fp32, BLOCK_SIZE=BLOCK_SIZE
    )

    if not out.is_contiguous():
        out.copy_(out_c)

    return out
