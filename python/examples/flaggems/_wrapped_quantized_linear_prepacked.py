"""``_wrapped_quantized_linear_prepacked`` port (quantized linear on the
FlagGems packed-weight format).

Port notes (triton-riscv, vs the upstream flag_gems file):
- The packer op ``_wrapped_linear_prepack`` is not part of this batch, so
  its ``unpack_linear_weight`` helper and the shared round-half-to-even
  helper are inlined verbatim from upstream's ``_wrapped_linear_prepack``.
- @libentry and the torch device context manager are stripped (repo
  conventions); kernels and wrapper arithmetic are upstream's verbatim.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_HEADER_BYTES = 16


def unpack_linear_weight(packed_weight, out_channel, in_channel=None):
    """Return views of the private packing format used by the paired op."""
    if packed_weight.dtype != torch.uint8 or packed_weight.ndim != 1:
        raise RuntimeError("packed_weight must be a one-dimensional uint8 tensor")
    if out_channel < 0:
        raise RuntimeError("out_channel must be non-negative")
    payload_bytes = packed_weight.numel() - _HEADER_BYTES - 4 * out_channel
    if payload_bytes < 0:
        raise RuntimeError("packed_weight has an invalid FlagGems linear layout")
    if out_channel == 0:
        if payload_bytes != 0:
            raise RuntimeError("packed_weight has an invalid FlagGems linear layout")
        K = 0 if in_channel is None else in_channel
    else:
        if payload_bytes % out_channel != 0:
            raise RuntimeError("packed_weight has an invalid FlagGems linear layout")
        K = payload_bytes // out_channel
        if in_channel is not None and K != in_channel:
            raise RuntimeError("input and packed weight dimensions do not match")
    metadata = packed_weight[:_HEADER_BYTES].view(torch.float32)
    bias_start = _HEADER_BYTES
    bias_end = bias_start + 4 * out_channel
    bias = packed_weight[bias_start:bias_end].view(torch.float32)
    weight = packed_weight[bias_end:].view(torch.int8).view(out_channel, K)
    return weight, metadata, bias


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
def _wrapped_quantized_linear_prepacked_empty_k_kernel(
    output,
    output_scale,
    output_zero_point,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    value = -tl.load(output_zero_point).to(tl.float32)
    value *= tl.load(output_scale).to(tl.float32)
    tl.store(output + offsets, value, mask=offsets < numel)


@triton.jit
def _wrapped_quantized_linear_prepacked_kernel(
    input,
    input_scale,
    input_zero_point,
    weight,
    weight_metadata,
    bias,
    output_scale,
    output_zero_point,
    output,
    M,
    N,
    K,
    stride_im,
    stride_ik,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    input_scale_value = tl.load(input_scale).to(tl.float32)
    input_zero_point_value = tl.load(input_zero_point).to(tl.int32)
    weight_scale_value = tl.load(weight_metadata).to(tl.float32)
    weight_zero_point_value = tl.load(weight_metadata + 1).to(tl.int32)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    input_sum = tl.zeros((BLOCK_M,), dtype=tl.int32)
    weight_sum = tl.zeros((BLOCK_N,), dtype=tl.int32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        current_k = k_block * BLOCK_K + offsets_k
        input_ptrs = (
            input + offsets_m[:, None] * stride_im + current_k[None, :] * stride_ik
        )
        input_mask = (offsets_m[:, None] < M) & (current_k[None, :] < K)
        input_values = tl.load(input_ptrs, mask=input_mask, other=0.0).to(tl.float32)
        quantized_input = (
            _round_half_to_even(input_values / input_scale_value)
            + input_zero_point_value
        )
        quantized_input = tl.minimum(tl.maximum(quantized_input, 0.0), 255.0).to(
            tl.uint8
        )
        shifted_input = (quantized_input.to(tl.int32) - 128).to(tl.int8)
        shifted_input = tl.where(input_mask, shifted_input, 0)

        weight_ptrs = weight + offsets_n[None, :] * K + current_k[:, None]
        weight_mask = (offsets_n[None, :] < N) & (current_k[:, None] < K)
        quantized_weight = tl.load(weight_ptrs, mask=weight_mask, other=0)

        accumulator += tl.dot(shifted_input, quantized_weight)
        input_sum += tl.sum(shifted_input.to(tl.int32), axis=1)
        weight_sum += tl.sum(quantized_weight.to(tl.int32), axis=0)

    shifted_input_zero_point = input_zero_point_value - 128
    accumulator -= weight_zero_point_value * input_sum[:, None]
    accumulator -= shifted_input_zero_point * weight_sum[None, :]
    accumulator += K * shifted_input_zero_point * weight_zero_point_value

    bias_values = tl.load(bias + offsets_n, mask=offsets_n < N, other=0.0)
    real_output = (
        accumulator.to(tl.float32) * input_scale_value * weight_scale_value
        + bias_values[None, :]
    )
    output_scale_value = tl.load(output_scale).to(tl.float32)
    output_zero_point_value = tl.load(output_zero_point).to(tl.float32)
    quantized_output = (
        _round_half_to_even(real_output / output_scale_value) + output_zero_point_value
    )
    quantized_output = tl.minimum(tl.maximum(quantized_output, 0.0), 255.0)
    dequantized_output = (
        quantized_output - output_zero_point_value
    ) * output_scale_value

    output_ptrs = output + offsets_m[:, None] * N + offsets_n[None, :]
    output_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(output_ptrs, dequantized_output, mask=output_mask)


def _wrapped_quantized_linear_prepacked(
    input,
    input_scale,
    input_zero_point,
    packed_weight,
    output_scale,
    output_zero_point,
    out_channel,
):
    logger.debug("GEMS _WRAPPED_QUANTIZED_LINEAR_PREPACKED")
    if input.dtype != torch.float32:
        raise RuntimeError(f"Quantize only works on Float Tensor, got {input.dtype}")
    if input.ndim < 2:
        raise RuntimeError(
            "The dimension of input tensor should be larger than or equal to 2"
        )
    if out_channel < 0:
        raise RuntimeError("out_channel must be non-negative")
    for name, parameter in (
        ("input_scale", input_scale),
        ("input_zero_point", input_zero_point),
        ("output_scale", output_scale),
        ("output_zero_point", output_zero_point),
    ):
        if parameter.numel() != 1:
            raise RuntimeError(f"{name} must contain one element")
        if parameter.device != input.device:
            raise RuntimeError(f"{name} must be on the input device")
    if packed_weight.device != input.device:
        raise RuntimeError("packed_weight must be on the input device")

    K = input.shape[-1]
    quantized_weight, weight_metadata, bias = unpack_linear_weight(
        packed_weight, out_channel, K
    )

    output_shape = (*input.shape[:-1], out_channel)
    output = torch.empty(output_shape, dtype=torch.float32, device=input.device)
    M = 1
    for dimension in input.shape[:-1]:
        M *= dimension
    if M == 0 or out_channel == 0:
        return output
    if K == 0:
        block_size = 256
        grid = (triton.cdiv(output.numel(), block_size),)
        _wrapped_quantized_linear_prepacked_empty_k_kernel[grid](
            output,
            output_scale,
            output_zero_point,
            output.numel(),
            BLOCK_SIZE=block_size,
        )
        return output

    input_2d = input.reshape(M, K)
    block_m = 32
    block_n = 64 if M >= 256 and out_channel >= 1024 else 32
    block_k = 32
    grid = (triton.cdiv(M, block_m), triton.cdiv(out_channel, block_n))
    _wrapped_quantized_linear_prepacked_kernel[grid](
        input_2d,
        input_scale,
        input_zero_point,
        quantized_weight,
        weight_metadata,
        bias,
        output_scale,
        output_zero_point,
        output,
        M,
        out_channel,
        K,
        input_2d.stride(0),
        input_2d.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return output
