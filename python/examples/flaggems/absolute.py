import torch
import triton
import triton.language as tl


@triton.jit
def _absolute_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    zero = x * 0
    is_neg = x < zero
    y = tl.where(is_neg, -x, x)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _absolute_complex_kernel(
    real_ptr,
    imag_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    re = tl.load(real_ptr + offsets, mask=mask)
    im = tl.load(imag_ptr + offsets, mask=mask)
    y = tl.sqrt(re * re + im * im)
    tl.store(out_ptr + offsets, y, mask=mask)


def absolute(input: torch.Tensor):
    x = input.contiguous()
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    if x.is_complex():
        ri = torch.view_as_real(x)
        # Feed the kernel unit-stride buffers.  StructuredToMemref currently
        # cannot lower two interleaved (stride-two) views of the same pointer.
        real = ri[..., 0].contiguous()
        imag = ri[..., 1].contiguous()
        out_dtype = x.real.dtype
        out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        _absolute_complex_kernel[grid](
            real, imag, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        return out
    else:
        out = torch.empty_like(x)
        _absolute_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out
