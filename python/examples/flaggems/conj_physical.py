import torch
import triton
import triton.language as tl


@triton.jit
def conj_physical_kernel(
    real_ptr,
    imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    real = tl.load(real_ptr + offsets, mask=mask)
    imag = tl.load(imag_ptr + offsets, mask=mask)

    tl.store(out_real_ptr + offsets, real, mask=mask)
    tl.store(out_imag_ptr + offsets, -imag, mask=mask)


def conj_physical(input):
    if not input.is_complex():
        return input

    n_elements = input.numel()
    real = input.real.contiguous()
    imag = input.imag.contiguous()
    output_real = torch.empty_like(real)
    output_imag = torch.empty_like(imag)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    conj_physical_kernel[grid](
        real,
        imag,
        output_real,
        output_imag,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return torch.complex(output_real, output_imag)
