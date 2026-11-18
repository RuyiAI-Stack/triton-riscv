import torch
import triton
import triton.language as tl

_WRAP_INTEGER_EXPONENT = tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 11)

_FALLBACK_DTYPES = (
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.bool,
    torch.int8,
    torch.uint8,
)


@triton.jit
def ldexp_inplace_kernel(
    x_ptr,
    e_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    e = tl.load(e_ptr + offsets, mask=mask, other=0)

    # scalbn via the IEEE-754 bit layout (exactly one rounding step); see
    # ldexp.py for the full derivation.
    u = x.to(tl.uint32, bitcast=True)
    uz = u & 0
    sign = u & (uz | 0x80000000)
    E = ((u >> 23) & 0xFF).to(tl.int32)
    M = u & (uz | 0x7FFFFF)

    is_special = (E == 255) | ((E == 0) & (M == 0))
    is_sub = (E == 0) & (M != 0)
    absx = (u & (uz | 0x7FFFFFFF)).to(tl.float32, bitcast=True)
    scaled = absx * 85070591730234615865843651857942052864.0
    sb = scaled.to(tl.uint32, bitcast=True)
    sE = ((sb >> 23) & 0xFF).to(tl.int32)
    sM = sb & (uz | 0x7FFFFF)

    M_eff = tl.where(is_sub, sM, M)
    e_use = tl.minimum(tl.maximum(e, -1000), 1000)
    t2 = tl.where(is_sub, sE - 253, E - 127) + e_use

    normal_bits = sign | (((t2 + 127) << 23).to(tl.uint32)) | M_eff
    over_bits = sign | (uz | 0x7F800000)

    sig24 = (M_eff | (uz | 0x800000)).to(tl.int32)
    shift = -126 - t2
    shift_c = tl.minimum(tl.maximum(shift, 1), 26)
    guard = (sig24 >> (shift_c - 1)) & 1
    sticky = (sig24 & ((1 << (shift_c - 1)) - 1)) != 0
    r = sig24 >> shift_c
    round_up = (guard == 1) & (sticky | ((r & 1) == 1))
    r = tl.where(round_up, r + 1, r)
    r_bits = tl.where(r == (1 << 23), uz | 0x800000, r.to(tl.uint32) & (uz | 0x7FFFFF))
    sub_bits = sign | r_bits

    bits = tl.where(t2 > 127, over_bits, tl.where(t2 >= -126, normal_bits, sub_bits))
    bits = tl.where(is_special, u, bits)
    tl.store(out_ptr + offsets, bits.to(tl.float32, bitcast=True), mask=mask)


@triton.jit
def ldexp_inplace_floatexp_kernel(
    x_ptr,
    e_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    e = tl.load(e_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, x * tl.exp2(e), mask=mask)


def ldexp_(self, other):
    if not isinstance(other, torch.Tensor):
        raise TypeError(
            f"ldexp(): argument 'other' (position 2) must be Tensor, "
            f"not {type(other).__name__}"
        )
    if not self.is_floating_point():
        # a float result cannot be cast back into an integer/bool self
        raise RuntimeError(
            f"ldexp can't be cast to the desired output type {self.dtype}"
        )
    if (
        self.is_complex()
        or other.is_complex()
        or self.dtype in _FALLBACK_DTYPES
        or other.dtype in _FALLBACK_DTYPES
    ):
        return torch.ops.aten.ldexp_(self, other)

    other = torch.broadcast_to(other, self.shape)
    if self.numel() == 0:
        return self

    n_elements = self.numel()
    x_c = self.contiguous().to(torch.float32)
    out = torch.empty(self.shape, dtype=torch.float32)
    grid = (triton.cdiv(n_elements, 1024),)
    if other.is_floating_point():
        e_c = other.contiguous().to(torch.float32)
        ldexp_inplace_floatexp_kernel[grid](x_c, e_c, out, n_elements, BLOCK_SIZE=1024)
    else:
        if _WRAP_INTEGER_EXPONENT:
            e_c = other.to(torch.int32)
        else:
            e_c = other.clamp(-(2**31), 2**31 - 1).to(torch.int32)
        e_c = e_c.contiguous()
        ldexp_inplace_kernel[grid](x_c, e_c, out, n_elements, BLOCK_SIZE=1024)

    # copy_ casts back to the self dtype and writes through non-contiguous
    # strides, so the view identity and dtype are preserved.
    self.copy_(out)
    return self
