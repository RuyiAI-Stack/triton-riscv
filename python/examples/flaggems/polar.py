import torch
import triton
import triton.language as tl


@triton.jit
def polar_kernel(
    abs_ptr,
    angle_ptr,
    real_ptr,
    imag_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    abs_val = tl.load(abs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    angle_val = tl.load(angle_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    real = abs_val * tl.cos(angle_val)
    imag = abs_val * tl.sin(angle_val)

    tl.store(real_ptr + offsets, real.to(real_ptr.dtype.element_ty), mask=mask)
    tl.store(imag_ptr + offsets, imag.to(imag_ptr.dtype.element_ty), mask=mask)


def polar(abs, angle):
    abs, angle = torch.broadcast_tensors(abs, angle)
    abs = abs.contiguous()
    angle = angle.contiguous()

    n_elements = abs.numel()
    real = torch.empty_like(abs)
    imag = torch.empty_like(abs)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    polar_kernel[grid](abs, angle, real, imag, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return torch.complex(real, imag)
