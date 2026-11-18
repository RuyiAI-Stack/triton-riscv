import torch
import triton
import triton.language as tl


@triton.jit
def chalf_real_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Real input -> complex32 output: real part = input, imaginary part = 0.
    # out_ptr is a view_as_real (float16) buffer with interleaved real/imag
    # slots.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    base = offsets * 2
    val = tl.load(in_ptr + offsets, mask=mask)
    zero = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    tl.store(out_ptr + base, val, mask=mask)
    tl.store(out_ptr + base + 1, zero, mask=mask)


@triton.jit
def chalf_complex_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Complex input -> complex32 output: cast the real/imag slots down to
    # float16. Both pointers are view_as_real buffers and n_elements counts
    # float slots.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, val, mask=mask)


def chalf(input, memory_format=None):
    BLOCK_SIZE = 1024

    if input.is_complex():
        # Resolve the conjugate bit so view_as_real reflects the logical
        # values.
        if input.is_conj():
            input = input.resolve_conj()
        src = input if input.is_contiguous() else input.contiguous()
        output = torch.empty(src.shape, dtype=torch.complex32, device=src.device)

        in_real = torch.view_as_real(src)
        out_real = torch.view_as_real(output)
        n_elements = in_real.numel()

        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        chalf_complex_kernel[grid](in_real, out_real, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return output

    # Real (or integer) input -> complex32 with zero imaginary part.
    src = input if input.is_contiguous() else input.contiguous()
    output = torch.empty(src.shape, dtype=torch.complex32, device=src.device)

    out_real = torch.view_as_real(output)
    n_elements = src.numel()

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    chalf_real_kernel[grid](src, out_real, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
