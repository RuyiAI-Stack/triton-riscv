import torch
import triton
import triton.language as tl

# torch >= 2.11 wraps integer exponents into int32 (ldexp(1.5, 2**32 + 5)
# acts as ldexp(1.5, 5)); older versions clamp them to the int32 range.
_WRAP_INTEGER_EXPONENT = tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 11)

# Routed to torch instead of the kernel: double precision, complex numbers and
# the 8-bit/bool dtypes whose conversion corners are not probed here.
_FALLBACK_DTYPES = (
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.bool,
    torch.int8,
    torch.uint8,
)


@triton.jit
def ldexp_kernel(
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

    # scalbn via the IEEE-754 bit layout: exactly one rounding step, so
    # deep-subnormal results match torch bit for bit (a x * exp2(e) product
    # double-rounds them).  NaN/inf keep their payload, +-0 keeps its sign.
    u = x.to(tl.uint32, bitcast=True)
    uz = u & 0
    sign = u & (uz | 0x80000000)
    E = ((u >> 23) & 0xFF).to(tl.int32)
    M = u & (uz | 0x7FFFFF)

    is_special = (E == 255) | ((E == 0) & (M == 0))

    # Subnormal x: scale by 2**126 (exact) to reveal the normal form, then
    # subtract both the bias and the scale from the exponent.  Large uint32
    # literals are built from uz so they stay unsigned (see nextafter.py).
    is_sub = (E == 0) & (M != 0)
    absx = (u & (uz | 0x7FFFFFFF)).to(tl.float32, bitcast=True)
    scaled = absx * 85070591730234615865843651857942052864.0
    sb = scaled.to(tl.uint32, bitcast=True)
    sE = ((sb >> 23) & 0xFF).to(tl.int32)
    sM = sb & (uz | 0x7FFFFF)

    M_eff = tl.where(is_sub, sM, M)

    # |e| beyond +-1000 only picks the overflow/underflow branch, so clamping
    # keeps the int32 arithmetic from wrapping for e near +-2**31.
    e_use = tl.minimum(tl.maximum(e, -1000), 1000)
    t2 = tl.where(is_sub, sE - 253, E - 127) + e_use

    normal_bits = sign | (((t2 + 127) << 23).to(tl.uint32)) | M_eff
    over_bits = sign | (uz | 0x7F800000)

    # Underflow: shift the 24-bit significand right with round-to-nearest-even
    # (guard + sticky); shift is clamped to 26 because everything past it
    # rounds to zero anyway and 32-bit shifts saturate.
    sig24 = (M_eff | (uz | 0x800000)).to(tl.int32)
    shift = -126 - t2
    shift_c = tl.minimum(tl.maximum(shift, 1), 26)
    guard = (sig24 >> (shift_c - 1)) & 1
    sticky = (sig24 & ((1 << (shift_c - 1)) - 1)) != 0
    r = sig24 >> shift_c
    round_up = (guard == 1) & (sticky | ((r & 1) == 1))
    r = tl.where(round_up, r + 1, r)
    # rounding carried into the implicit bit -> smallest normal
    r_bits = tl.where(r == (1 << 23), uz | 0x800000, r.to(tl.uint32) & (uz | 0x7FFFFF))
    sub_bits = sign | r_bits

    bits = tl.where(t2 > 127, over_bits, tl.where(t2 >= -126, normal_bits, sub_bits))
    bits = tl.where(is_special, u, bits)
    tl.store(out_ptr + offsets, bits.to(tl.float32, bitcast=True), mask=mask)


@triton.jit
def ldexp_floatexp_kernel(
    x_ptr,
    e_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # A float exponent degrades ldexp to x * 2**e (torch does the same).
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    e = tl.load(e_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, x * tl.exp2(e), mask=mask)


def _exponent_to_int32(other):
    if _WRAP_INTEGER_EXPONENT:
        return other.to(torch.int32)
    return other.clamp(-(2**31), 2**31 - 1).to(torch.int32)


def _result_dtype(self, other):
    if self.is_floating_point():
        return self.dtype
    # INT_TO_FLOAT promotion: integer inputs yield float32.
    return torch.float32


def ldexp(self, other):
    if not isinstance(other, torch.Tensor):
        raise TypeError(
            f"ldexp(): argument 'other' (position 2) must be Tensor, "
            f"not {type(other).__name__}"
        )
    if (
        self.is_complex()
        or other.is_complex()
        or self.dtype in _FALLBACK_DTYPES
        or other.dtype in _FALLBACK_DTYPES
    ):
        return torch.ldexp(self, other)

    result_dtype = _result_dtype(self, other)
    self, other = torch.broadcast_tensors(self, other)
    n_elements = self.numel()
    out = torch.empty(self.shape, dtype=torch.float32)
    if n_elements == 0:
        return out.to(result_dtype)

    x_c = self.contiguous().to(torch.float32)
    if other.is_floating_point():
        e_c = other.contiguous().to(torch.float32)
        grid = (triton.cdiv(n_elements, 1024),)
        ldexp_floatexp_kernel[grid](x_c, e_c, out, n_elements, BLOCK_SIZE=1024)
    else:
        e_c = _exponent_to_int32(other).contiguous()
        grid = (triton.cdiv(n_elements, 1024),)
        ldexp_kernel[grid](x_c, e_c, out, n_elements, BLOCK_SIZE=1024)
    return out.to(result_dtype)


def ldexp_out(self, other, *, out):
    result = ldexp(self, other)
    if not torch.can_cast(result.dtype, out.dtype):
        raise RuntimeError(
            f"result type {result.dtype} can't be cast to the desired "
            f"output type {out.dtype}"
        )
    out.resize_(result.shape)
    out.copy_(result)
    return out
