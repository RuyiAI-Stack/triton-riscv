import torch
import triton
import triton.language as tl

# _conj_copy: aten::_conj_copy as a value copy. The (contiguous) complex
# tensor is reinterpreted as its flat float dtype; conjugation keeps even
# lanes (real) and negates odd lanes (imag). A pre-existing conj bit is
# resolved first so conj(conj(x)) == x.

_CONJ_BLOCK = 1024


@triton.jit
def _conj_copy_kernel(in_ptr, out_ptr, n_floats, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_floats
    vals = tl.load(in_ptr + offsets, mask=mask)
    is_imag = (offsets % 2) == 1
    out_vals = tl.where(is_imag, -vals, vals)
    tl.store(out_ptr + offsets, out_vals, mask=mask)


def _float_dtype(x: torch.Tensor) -> torch.dtype:
    if x.dtype == torch.complex32:
        return torch.float16
    if x.dtype == torch.complex64:
        return torch.float32
    if x.dtype == torch.complex128:
        return torch.float64
    raise RuntimeError("_conj_copy only supports complex tensors")


def _conj_copy(input: torch.Tensor) -> torch.Tensor:
    """Fresh conjugated copy of a complex tensor (interleaved lanes negated on imag)."""
    if not input.is_complex():
        raise RuntimeError("_conj_copy only supports complex tensors")
    if input.is_conj():
        input = input.resolve_conj()
    src = input if input.is_contiguous() else input.contiguous()
    output = torch.empty_like(src)
    float_dt = _float_dtype(src)
    in_float = src.reshape(-1).view(float_dt)
    out_float = output.reshape(-1).view(float_dt)
    n_floats = in_float.numel()
    grid = (triton.cdiv(n_floats, _CONJ_BLOCK),)
    _conj_copy_kernel[grid](in_float, out_float, n_floats, BLOCK_SIZE=_CONJ_BLOCK)
    return output


def _conj_copy_out(input: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """``_conj_copy`` writing into ``out`` (dtype/device/numel checked)."""
    if not input.is_complex():
        raise RuntimeError("_conj_copy only supports complex tensors")
    if out.dtype != input.dtype:
        raise RuntimeError("_conj_copy.out: dtype of input and output must match.")
    if out.device != input.device:
        raise RuntimeError(
            "_conj_copy.out: input and output must be on the same device."
        )
    if out.numel() != input.numel():
        raise RuntimeError(
            "_conj_copy.out: input and output must have the same number of elements."
        )
    if input.is_conj():
        input = input.resolve_conj()
    src = input if input.is_contiguous() else input.contiguous()
    dst = out if out.is_contiguous() else out.contiguous()
    float_dt = _float_dtype(src)
    in_float = src.reshape(-1).view(float_dt)
    out_float = dst.reshape(-1).view(float_dt)
    n_floats = in_float.numel()
    grid = (triton.cdiv(n_floats, _CONJ_BLOCK),)
    _conj_copy_kernel[grid](in_float, out_float, n_floats, BLOCK_SIZE=_CONJ_BLOCK)
    if dst.data_ptr() != out.data_ptr():
        out.copy_(dst)
    return out
