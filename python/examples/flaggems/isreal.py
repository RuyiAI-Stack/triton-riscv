import torch
import triton
import triton.language as tl

_REAL_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
)
_COMPLEX_DTYPES = (torch.complex64, torch.complex128)


@triton.jit
def isreal_true_kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Store True for every element; every element of a real tensor is real.

    The input is never read: ``isreal`` is unconditionally True for real
    dtypes, including NaN and infinity, so this path is a pure fill.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, 1, mask=mask)


@triton.jit
def isreal_complex_kernel(
    in_real_ptr, out_ptr, n_elements, IS_FP64: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """Test the imaginary component of each complex element for an exact zero.

    Complex storage interleaves ``(real, imag)`` pairs of the underlying
    floating-point type.  The imaginary part is read with a stride-2 load
    (coalesces into the same transactions as the interleaved pair load), and
    it is zero (including a negative zero) exactly when the element is real.
    ``imag == 0.0`` alone is False for a negative zero on this target, so the
    sign bit is cleared before the compare, exactly as IEEE requires.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    imag = tl.load(in_real_ptr + offsets * 2 + 1, mask=mask, other=0.0)
    if IS_FP64:
        bits = imag.to(tl.int64, bitcast=True) & 0x7FFFFFFFFFFFFFFF
    else:
        bits = imag.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    tl.store(out_ptr + offsets, bits == 0, mask=mask)


def isreal(input: torch.Tensor) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"isreal expects a Tensor, got {type(input).__name__}")
    if input.dtype not in _REAL_DTYPES + _COMPLEX_DTYPES:
        raise TypeError(
            "isreal only supports floating point, complex, integral and bool "
            f"inputs, got {input.dtype}"
        )

    out = torch.empty(input.size(), dtype=torch.bool, device=input.device)
    n_elements = input.numel()
    if n_elements == 0:
        return out

    if input.dtype in _COMPLEX_DTYPES:
        # A lazy conjugate or negation only flips the sign of the imaginary
        # component, which cannot turn a zero into a non-zero, so both bits may
        # be ignored.  They must still be resolved before taking the storage
        # view, since torch.view_as_real rejects lazy conjugate/negation
        # tensors.
        if input.is_conj():
            input = input.resolve_conj()
        if input.is_neg():
            input = input.resolve_neg()
        # The kernel walks the interleaved storage linearly, so non-contiguous
        # inputs need to be materialized first.
        if not input.is_contiguous():
            input = input.contiguous()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        isreal_complex_kernel[grid](
            torch.view_as_real(input),
            out,
            n_elements,
            IS_FP64=input.dtype == torch.complex128,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # The real path never reads the input, so the only cost is the bool
        # store; a wide block minimizes the number of programs for this
        # pure-fill kernel.
        BLOCK_SIZE = 4096
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        isreal_true_kernel[grid](out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
