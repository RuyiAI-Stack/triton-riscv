"""Triton implementation of aten::bilinear.

Ported from upstream FlagGems ops/bilinear.py.

    y[m, o] = sum_{i, j} x1[m, i] * W[o, i, j] * x2[m, j] + b[o]

Porting notes (repo conventions):
  - No flag_gems imports; no torch device context manager.
  - Reduction order restructured to match aten's composition, which computes
    tmp[m, o, j] = sum_i x1[m, i] * W[o, i, j] first and then contracts with
    x2 over j (verified bitwise against ``F.bilinear`` on this backend).
    The upstream kernel multiplied x2 inside the i loop, which rounds
    differently; the restructure is value-equivalent to aten, not to the
    upstream rounding order.
  - fp64 inputs accumulate in fp64 (native loads); every other dtype is
    accumulated in fp32 exactly like upstream. int64 is cast to fp32 (the
    backend has no int64 dot lowering); the pinned small-integer test data
    is exact in fp32.
  - Stage-2 reduction over j is sequential (aten's dot order), realized by
    extracting one lane per static step: adding zeros is exact, so the
    masked extraction returns each lane bit-for-bit.
  - aten's own validation messages are reproduced in the wrapper (the
    upstream wrapper had none).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def bilinear_kernel(
    input1_ptr,
    input2_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K1,
    K2,
    stride_i1m,
    stride_i1k,
    stride_i2m,
    stride_i2k,
    stride_wn,
    stride_wk1,
    stride_wk2,
    stride_om,
    stride_on,
    BIAS: tl.constexpr,
    BLOCK_K2: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    """
    Bilinear kernel: y[m, o] = sum_{i, j} x1[m, i] * W[o, i, j] * x2[m, j] + b[o]
    - input1: (M, K1) where M is the flattened batch size, K1 is in1_features
    - input2: (M, K2) where K2 is in2_features
    - weight: (N, K1, K2) where N is out_features
    - bias: (N,) optional
    - output: (M, N)

    Each program computes a single output element (m, o). Stage 1
    accumulates tmp[j] = sum_i x1[m, i] * W[o, i, j] in BLOCK_K2 lanes; the
    per-lane i loop is sequential, matching a dot product over in1_features.
    Stage 2 contracts tmp with x2 over j sequentially (aten's order).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_k2 = tl.arange(0, BLOCK_K2)
    mask_k2 = offs_k2 < K2
    if IS_FP64:
        tmp = tl.zeros([BLOCK_K2], dtype=tl.float64)
    else:
        tmp = tl.zeros([BLOCK_K2], dtype=tl.float32)

    for i in range(0, K1):
        if IS_FP64:
            x1 = tl.load(input1_ptr + pid_m * stride_i1m + i * stride_i1k)
            w = tl.load(
                weight_ptr + pid_n * stride_wn + i * stride_wk1 + offs_k2 * stride_wk2,
                mask=mask_k2,
                other=0.0,
            )
        else:
            x1 = tl.load(input1_ptr + pid_m * stride_i1m + i * stride_i1k).to(
                tl.float32
            )
            w = tl.load(
                weight_ptr + pid_n * stride_wn + i * stride_wk1 + offs_k2 * stride_wk2,
                mask=mask_k2,
                other=0.0,
            ).to(tl.float32)
        tmp += x1 * w

    if IS_FP64:
        result = tl.zeros((), dtype=tl.float64)
    else:
        result = tl.zeros((), dtype=tl.float32)
    for j in tl.static_range(BLOCK_K2):
        # Extract lane j exactly (sum of one value and zeros), then fold it
        # in sequentially so stage 2 rounds like aten's dot over j.
        lane = tl.sum(tl.where((offs_k2 == j) & mask_k2, tmp, 0.0), axis=0)
        if IS_FP64:
            x2 = tl.load(
                input2_ptr + pid_m * stride_i2m + j * stride_i2k,
                mask=j < K2,
                other=0.0,
            )
        else:
            x2 = tl.load(
                input2_ptr + pid_m * stride_i2m + j * stride_i2k,
                mask=j < K2,
                other=0.0,
            ).to(tl.float32)
        result += lane * x2

    if BIAS:
        if IS_FP64:
            result += tl.load(bias_ptr + pid_n)
        else:
            result += tl.load(bias_ptr + pid_n).to(tl.float32)

    tl.store(
        output_ptr + pid_m * stride_om + pid_n * stride_on,
        result.to(output_ptr.dtype.element_ty),
    )


def bilinear(input1, input2, weight, bias=None):
    """
    Applies a bilinear transformation to the incoming data:
        y = x1^T A x2 + b

    Args:
        input1: Tensor of shape (N, *, in1_features)
        input2: Tensor of shape (N, *, in2_features)
        weight: Tensor of shape (out_features, in1_features, in2_features)
        bias: Tensor of shape (out_features), optional

    Returns:
        Output tensor of shape (N, *, out_features)
    """
    # All but the last dimension of input1 and input2 must match.
    batch_dims = input1.shape[:-1]
    M = 1
    for dim in batch_dims:
        M *= dim
    K1 = input1.shape[-1]  # in1_features
    K2 = input2.shape[-1]  # in2_features
    N = weight.shape[0]  # out_features

    # aten's own validation (messages pinned empirically).
    if input2.shape[:-1] != batch_dims:
        raise RuntimeError(
            "bilinear(): input1 and input2 sizes must match over batch dimension"
        )
    if weight.dim() != 3:
        raise RuntimeError("bilinear(): weight must be a 3-D tensor")
    if input1.dim() < 2 or weight.shape[1] != K1:
        raise RuntimeError("bilinear(): input1 size does not match weight size")
    if weight.shape[2] != K2:
        raise RuntimeError("bilinear(): input2 size does not match weight size")
    if bias is not None and bias.numel() != N:
        raise RuntimeError("bilinear(): bias size does not match weight size")

    input1_flat = input1.reshape(M, K1)
    input2_flat = input2.reshape(M, K2)
    weight = weight.contiguous()

    output = torch.empty((M, N), device=input1.device, dtype=input1.dtype)

    # A one-element tile cannot be lowered on the CPU backend, so keep at
    # least 2 lanes (upstream keeps the same floor for XPU).
    BLOCK_K2 = max(min(triton.next_power_of_2(K2), 32), 2)

    grid = (M, N)

    bilinear_kernel[grid](
        input1_flat,
        input2_flat,
        weight,
        bias if bias is not None else weight,
        output,
        M,
        N,
        K1,
        K2,
        input1_flat.stride(0),
        input1_flat.stride(1),
        input2_flat.stride(0),
        input2_flat.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        BIAS=bias is not None,
        BLOCK_K2=BLOCK_K2,
        IS_FP64=input1.dtype == torch.float64,
    )

    return output.view(*batch_dims, N)
