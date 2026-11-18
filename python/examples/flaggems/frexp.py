import torch
import triton
import triton.language as tl


@triton.jit
def frexp_kernel(
    x_ptr,
    mantissa_ptr,
    exponent_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Decompose x = mantissa * 2**exponent with |mantissa| in [0.5, 1.0).
    #
    # Special cases: frexp(+0) = (+0, 0), frexp(-0) = (-0, 0),
    # frexp(+-inf) = (+-inf, 0), frexp(nan) = (nan, 0).
    #
    # The computation runs in float32 (exact for fp16/bf16/fp32 inputs) and
    # decomposes via the IEEE-754 bit layout instead of log2/exp2
    # arithmetic: log2 rounds to its own precision, so for values one ULP
    # below a power of two it returns the exact integer exponent and
    # floor(log2(x)) + 1 lands one octave too high.  The bit extraction is
    # exact for every finite value.
    x_fp32 = x.to(tl.float32)
    abs_x = tl.abs(x_fp32)

    # NaN is the only floating-point value that is not equal to itself.
    is_nan = x_fp32 != x_fp32
    is_inf = abs_x == float("inf")
    is_zero = x_fp32 == 0.0
    is_special = is_nan | is_inf | is_zero

    # Normal numbers: reuse the biased exponent field and splice the
    # payload bits into a value in [0.5, 1.0):
    #     exponent = exp_bits - 126
    #     mantissa_bits = (bits & 0x007FFFFF) | 0x3F000000
    # (bits comes from abs_x, so the sign bit is clear and the mask stays a
    # positive int32 constant.)
    bits = abs_x.to(tl.int32, bitcast=True)
    exp_bits = (bits >> 23) & 0xFF
    is_subnormal = exp_bits == 0

    # Subnormals (and zero) have no implicit leading 1.  Scale by 2**126 to
    # normalize, then subtract the scale from the exponent afterwards.
    scaled = tl.where(
        is_special, abs_x, abs_x * 85070591730234615865843651857942052864.0
    )
    scaled_bits = scaled.to(tl.int32, bitcast=True)
    scaled_exp_bits = (scaled_bits >> 23) & 0xFF
    scaled_mantissa_bits = (scaled_bits & 0x007FFFFF) | 0x3F000000

    mantissa_bits = tl.where(
        is_subnormal, scaled_mantissa_bits, (bits & 0x007FFFFF) | 0x3F000000
    )
    exponent = tl.where(is_subnormal, scaled_exp_bits - 126 - 126, exp_bits - 126)
    abs_mantissa = mantissa_bits.to(tl.float32, bitcast=True)
    mantissa = tl.where(x_fp32 < 0.0, -abs_mantissa, abs_mantissa)

    # Preserve the original value (incl. the sign of -0.0) for specials;
    # special values must have an exponent of zero.
    mantissa = tl.where(is_special, x_fp32, mantissa)
    exponent = tl.where(is_special, 0, exponent)

    # The mantissa uses the input dtype, the exponent is always int32.
    tl.store(mantissa_ptr + offsets, mantissa.to(x.dtype), mask=mask)
    tl.store(exponent_ptr + offsets, exponent, mask=mask)


def frexp(A):
    if not A.is_floating_point():
        raise RuntimeError(
            f"frexp(): expected a floating-point tensor, but got {A.dtype}"
        )

    # The decomposition converts values to float32 internally, so it cannot
    # correctly preserve the precision and range of float64 inputs.
    if A.dtype == torch.float64:
        raise RuntimeError("FlagGems frexp currently does not support float64")

    x = A.contiguous()
    n_elements = x.numel()

    # frexp returns tensors with different dtypes, so both outputs are
    # preallocated explicitly: the mantissa keeps the input dtype and the
    # exponent is int32.
    mantissa = torch.empty_like(x)
    exponent = torch.empty_like(x, dtype=torch.int32)
    if n_elements == 0:
        return mantissa, exponent

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    frexp_kernel[grid](x, mantissa, exponent, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return mantissa, exponent
