"""quantize_per_channel - per-channel affine quantization.

Port notes (triton-riscv, vs the upstream flag_gems file):
- flag_gems.runtime / libentry / tl_extra_shim are inlined: the backend
  lowers no libdevice externs, so ``nearbyint`` is a floor-based
  round-half-to-even helper and the reciprocal uses ``tl.math.div_rn``.
- The kernel arithmetic follows the CPU aten oracle, which was probed to
  compute, in fp32, ``rint(x * (1.f/scale)) + zero_point`` for the affine
  scheme (the fp32 reciprocal-multiply; a true division flips crafted tie
  values) and ``rint(x * (1.f/scale) + zero_point)`` for the float-qparams
  scheme, where the zero point is added *before* rounding.  The float branch
  launches with ``enable_fp_fusion=False`` so the mul+add feeding the rounding
  keeps aten's two roundings instead of a fused FMA.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# Quantized integer ranges per torch quantized dtype:
#   quint8  -> uint8  [0, 255]
#   qint8   -> int8   [-128, 127]
#   qint32  -> int32  [-2147483648, 2147483647]
# ``zero_points`` is the integer zero-point dtype each scheme stores; the float
# scheme keeps its zero-points in float32.
_QRANGE = {
    torch.quint8: (torch.uint8, 0, 255),
    torch.qint8: (torch.int8, -128, 127),
    torch.qint32: (torch.int32, -2147483648, 2147483647),
}


def _scalar_type_name(dtype):
    """Return the C++ type name PyTorch uses in its error messages."""
    return {
        torch.float16: "Half",
        torch.bfloat16: "BFloat16",
        torch.float64: "Double",
        torch.float32: "Float",
        torch.int8: "Char",
        torch.uint8: "Byte",
        torch.int16: "Short",
        torch.int32: "Int",
        torch.int64: "Long",
        torch.bool: "Bool",
        torch.quint8: "QUInt8",
        torch.qint8: "QInt8",
        torch.qint32: "QInt32",
        torch.quint4x2: "QUInt4x2",
    }.get(dtype, str(dtype))


@triton.jit
def _round_half_to_even(x):
    floor_x = tl.floor(x)
    fraction = x - floor_x
    floor_is_even = (floor_x % 2.0) == 0.0
    rounded = tl.where(
        fraction > 0.5,
        floor_x + 1.0,
        tl.where((fraction < 0.5) | floor_is_even, floor_x, floor_x + 1.0),
    )
    # floor() saturates non-finite inputs to huge finite values on some
    # backends (riscv64 CPU), which would round NaN/inf to a clamp bound.
    # Detect them from the fp32 exponent bits rather than by float comparison
    # so the passthrough does not depend on comparison semantics.
    abs_bits = x.to(tl.float32).to(tl.int32, bitcast=True) & 0x7FFFFFFF
    return tl.where(abs_bits >= 0x7F800000, x, rounded)


@triton.jit
def quantize_per_channel_kernel(
    x_ptr,
    scales_ptr,
    zero_points_ptr,
    out_ptr,
    n_elements,
    stride_axis,
    shape_axis,
    q_min: tl.constexpr,
    q_max: tl.constexpr,
    USE_FLOAT_ZERO_POINT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Channel index along `axis` for a contiguous (or axis-aligned) tensor:
    #   axis_coord = (flat_index // stride_axis) % shape_axis
    # Covers ``stride_axis == 1`` too (``// 1`` is the identity).  Do NOT
    # branch on the runtime scalar here: a runtime-scalar ``if`` chain
    # miscompiles on this backend into an out-of-bounds write (segfault).
    axis_coord = (offsets // stride_axis) % shape_axis

    scale = tl.load(scales_ptr + axis_coord, mask=mask, other=1.0)

    if USE_FLOAT_ZERO_POINT:
        # per_channel_affine_float_qparams: the zero point is added *before*
        # rounding, all in fp32 (probed against the CPU aten oracle).
        zero_point_f = tl.load(zero_points_ptr + axis_coord, mask=mask, other=0.0)
        inv_scale = tl.math.div_rn(1.0, scale.to(tl.float32))
        q = _round_half_to_even(x_val * inv_scale + zero_point_f)
    else:
        # per_channel_affine: the zero-point is a genuine integer, stored as
        # int64 and validated on the host against the target dtype's range
        # (check_zero_points_cuda).  The scaled value is rounded *first* and
        # the zero-point added after, so the addition sits outside the
        # rounding (probed against the CPU aten oracle).
        zero_point_i64 = tl.load(zero_points_ptr + axis_coord, mask=mask, other=0)
        inv_scale = tl.math.div_rn(1.0, scale.to(tl.float32))
        q = _round_half_to_even(x_val * inv_scale) + zero_point_i64.to(tl.float32)

    q = tl.clamp(q, q_min, q_max)
    q = q.to(out_ptr.dtype.element_ty)

    tl.store(out_ptr + offsets, q, mask=mask)


def _is_float_qparams(scales, zero_points):
    """Whether this qparam pair selects ``per_channel_affine_float_qparams``.

    Mirrors ``make_per_channel_affine_quantizer``, which picks the float scheme
    when (and only when) the zero-points are floating point.
    """
    return zero_points.dtype.is_floating_point


def _validate_qparams(input, scales, zero_points, axis, dtype):
    """Validate the input contract the way native ``make_per_channel_affine_quantizer`` does.

    Every check mirrors an ATen assertion, and it runs before the kernel is
    launched so a malformed qparam vector can never be read out of bounds by
    the kernel's per-element ``tl.load``.
    """
    if dtype not in _QRANGE:
        raise NotImplementedError(
            f'"quantize_tensor_per_channel_affine" not implemented for '
            f"'{_scalar_type_name(dtype)}'"
        )

    # Device consistency first -- it is what ATen's argument checker reports
    # before any semantic validation, for both qparam vectors. Native raises
    # instead of silently transferring, so a CPU qparam tensor never reaches
    # the kernel through an implicit H2D copy.
    for name, param in (("scales", scales), ("zero_points", zero_points)):
        if param.device != input.device:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got {name} is on "
                f"{param.device}, different from other tensors on {input.device} "
                "(when checking argument in method wrapper_CUDA__quantize_per_channel)"
            )

    # checkFloatTensor: only float32 inputs reach the native per-channel kernels.
    # The two quantizers phrase this differently -- the float-qparams quantizer
    # checks inside its own ``quantize`` ("Quantize only works on Float Tensor"),
    # the affine one inside the native kernel ("... expects a Float Tensor").
    if input.dtype != torch.float32:
        got = _scalar_type_name(input.dtype)
        if _is_float_qparams(scales, zero_points):
            raise RuntimeError(f"Quantize only works on Float Tensor, got {got}")
        raise RuntimeError(
            f"quantize_tensor_per_channel_affine expects a Float Tensor, got {got}"
        )

    # checkPerChannelParamDims: both vectors must be 1-D with equal length.
    if scales.dim() != 1:
        raise RuntimeError("scale tensor must have dimension 1")
    if zero_points.dim() != 1:
        raise RuntimeError("zero_points tensor must have dimension 1")
    if scales.numel() != zero_points.numel():
        raise RuntimeError("number of elements in scales and zero_points must match")

    # make_per_channel_affine_quantizer: scales are always floating point, and
    # a floating-point zero_points selects the *_float_qparams scheme.
    if not scales.dtype.is_floating_point:
        raise RuntimeError("scale tensor must be floating point")

    # ``axis`` is a Python int in the aten schema, so ``input.size(axis)`` below
    # is safe once the range check has run. Negative axes are rejected rather
    # than accepted, matching ATen's ``0 <= axis && axis < dim``.
    if not (0 <= axis < input.dim()):
        scheme = "float qparams" if _is_float_qparams(scales, zero_points) else "affine"
        raise RuntimeError(
            f"Channel axis out of range in per channel {scheme} quantization. "
            f"Got: {axis}Expected: [0, {input.dim()})"
        )

    n_channels = input.size(axis)
    if scales.numel() != n_channels:
        raise RuntimeError(
            f"length of scales must equal to channel, expected {n_channels} got, "
            f"{scales.numel()}"
        )
    if zero_points.numel() != n_channels:
        raise RuntimeError(
            f"length of zero_points must equal to channel expected {n_channels} got, "
            f"{zero_points.numel()}"
        )

    return _is_float_qparams(scales, zero_points)


def _check_zero_points_on_host(zero_points, q_min, q_max, use_float_zero_point):
    """Validate the zero-point range on the host, mirroring ``check_zero_points_cuda``.

    ATen performs this check inside the CUDA kernel after an explicit device
    synchronization, so the failure is a catchable synchronous RuntimeError
    and the CUDA context survives. The previous device-side PTX ``trap`` turned
    the same input error into an asynchronous, context-killing fault instead.

    ATen checks the lower bound first: a vector that violates both bounds
    reports "below lower bound".
    """
    scheme = "float_qparams" if use_float_zero_point else "affine"
    prefix = f"quantize_tensor_per_channel_{scheme}_cuda"
    if zero_points.numel() == 0:
        return
    zp_min = zero_points.min().item()
    if zp_min < q_min:
        raise RuntimeError(f"{prefix}zero_point is below lower bound.")
    zp_max = zero_points.max().item()
    if zp_max > q_max:
        raise RuntimeError(f"{prefix}zero_point is above upper bound.")


def _quantize_per_channel_impl(input, scales, zero_points, axis, dtype):
    use_float_zero_point = _validate_qparams(input, scales, zero_points, axis, dtype)
    int_dtype, q_min, q_max = _QRANGE[dtype]

    # The quantization params are 1-D vectors of length ``input.size(axis)``.
    # They are already guaranteed to be on the input's device by
    # ``_validate_qparams`` -- no implicit transfer happens here.
    #
    # Integer schemes store scales as fp64 and zero-points as int64, and the
    # float scheme keeps both in fp32 -- the dtypes the two native quantizers
    # use internally. The kernel branches on ``USE_FLOAT_ZERO_POINT``, so the
    # two precisions never mix.
    input = input.contiguous()
    # The zero-point range check syncs on the original dtype values, before
    # any conversion, like ATen's ``check_zero_points_cuda``.
    _check_zero_points_on_host(zero_points, q_min, q_max, use_float_zero_point)
    if use_float_zero_point:
        scales_k = scales.to(torch.float32).contiguous()
        zero_points_k = zero_points.to(torch.float32).contiguous()
    else:
        scales_k = scales.to(torch.float64).contiguous()
        zero_points_k = zero_points.to(torch.int64).contiguous()

    shape_axis = input.shape[axis]
    stride_axis = input.stride(axis)

    int_out = torch.empty_strided(
        input.shape,
        input.stride(),
        dtype=int_dtype,
        device=input.device,
    )

    n_elements = input.numel()
    if n_elements > 0:
        # Power-of-2 block size for coalesced element-wise access.
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        quantize_per_channel_kernel[grid](
            input,
            scales_k,
            zero_points_k,
            int_out,
            n_elements,
            stride_axis,
            shape_axis,
            q_min,
            q_max,
            USE_FLOAT_ZERO_POINT=use_float_zero_point,
            BLOCK_SIZE=BLOCK_SIZE,
            enable_fp_fusion=False,
        )

    # ``torch._make_per_channel_quantized_tensor`` copies the integer storage, so
    # it must run *after* the kernel has written ``int_out``, and it infers the
    # qscheme from the zero-point dtype -- exactly how the native quantizer
    # factory picks between per_channel_affine and
    # per_channel_affine_float_qparams.
    return torch._make_per_channel_quantized_tensor(
        int_out, scales_k, zero_points_k, axis
    )


def quantize_per_channel(input, scales, zero_points, axis, dtype):
    logger.debug("GEMS QUANTIZE_PER_CHANNEL")
    return _quantize_per_channel_impl(input, scales, zero_points, axis, dtype)


def quantize_per_channel_out(input, scales, zero_points, axis, dtype, *, out=None):
    logger.debug("GEMS QUANTIZE_PER_CHANNEL_OUT")
    if out is None:
        return _quantize_per_channel_impl(input, scales, zero_points, axis, dtype)
    return _quantize_per_channel_out_impl(input, scales, zero_points, axis, dtype, out)


def _quantize_per_channel_out_impl(input, scales, zero_points, axis, dtype, out):
    # ``out`` must be a quantized tensor of the requested dtype: the
    # autogenerated redispatch wrapper (``check_out_type``) rejects anything
    # else before the operator body runs.
    if not out.is_quantized or out.dtype != dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {dtype}, but got {out.dtype} instead"
        )
    # The native out= path copies the freshly-quantized tensor into ``out`` with
    # ``copy_``, which for a pair of quantized tensors requires the same qscheme
    # and then adopts the source quantizer. A float-qparams result therefore
    # cannot land in an integer-qparams ``out`` (and vice versa).
    float_result = _is_float_qparams(scales, zero_points)
    float_out = out.qscheme() == torch.per_channel_affine_float_qparams
    if float_result != float_out:
        raise RuntimeError("Quantized Copy only works with same qscheme")

    if tuple(out.shape) != tuple(input.shape):
        # ATen warns and resizes a non-empty mismatched out. Resizing a
        # per-channel quantized tensor is not implemented for this backend, so
        # only the exact-shape contract can be served; report the mismatch
        # rather than silently reinterpreting the destination storage.
        raise RuntimeError(
            f"The size of tensor a {tuple(out.shape)} must match the size of "
            f"tensor b {tuple(input.shape)} at non-singleton dimension 0"
        )

    result = _quantize_per_channel_impl(input, scales, zero_points, axis, dtype)
    # ``copy_`` replaces the destination's integer storage and adopts the
    # source scale/zero-point/axis, which is what the native out= does.
    out.copy_(result)
    return out
