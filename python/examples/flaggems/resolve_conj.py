import torch
import triton
import triton.language as tl


@triton.jit
def copy_complex_components_kernel(
    real_ptr,
    imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(
        out_real_ptr + offsets,
        tl.load(real_ptr + offsets, mask=mask),
        mask=mask,
    )
    tl.store(
        out_imag_ptr + offsets,
        tl.load(imag_ptr + offsets, mask=mask),
        mask=mask,
    )


def resolve_conj(A: torch.Tensor):
    if not A.is_conj():
        return A
    if not A.is_complex():
        return A.clone()

    # Contiguous component views materialize the logical value of a conjugate
    # view, so reassembling them clears the conjugate bit without interleaved
    # complex pointer arithmetic in Triton.
    real = A.real.contiguous()
    imag = A.imag.contiguous()
    out_real = torch.empty_like(real)
    out_imag = torch.empty_like(imag)
    n_elements = A.numel()
    BLOCK_SIZE = 1024
    copy_complex_components_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        real,
        imag,
        out_real,
        out_imag,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return torch.complex(out_real, out_imag)
