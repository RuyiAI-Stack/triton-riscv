import os
from contextlib import contextmanager
from math import isfinite
from pathlib import Path
from tempfile import mkdtemp

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver


def _activate_cpu_driver():
    triton.runtime.driver.set_active(CPUDriver())


@contextmanager
def _use_cpu_driver():
    previous_driver = getattr(triton.runtime.driver, "_active", None)
    _activate_cpu_driver()
    try:
        yield
    finally:
        if previous_driver is None:
            # Restore Triton's cold-start state without forcing device discovery.
            triton.runtime.driver._active = None
        else:
            triton.runtime.driver.set_active(previous_driver)





def make_3d_for_bn(x):
    if x.ndim == 2:
        return x.unsqueeze(-1)
    if x.ndim >= 4:
        return x.flatten(2, -1)
    return x


@triton.jit
def batch_norm_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    running_mean_pointer,
    running_var_pointer,
    batch_dim,
    spatial_dim,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    output_batch_stride,
    output_feat_stride,
    output_spatial_stride,
    momentum,
    eps,
    is_train: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    if is_train:
        sum_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        sq_sum_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
                spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
                spatial_mask = spatial_offset < spatial_dim
                mask = batch_mask[:, None] & spatial_mask[None, :]

                curr_input_pointer = (
                    input_pointer
                    + input_feat_stride * feat_pid
                    + input_batch_stride * batch_offset[:, None]
                    + input_spatial_stride * spatial_offset[None, :]
                )
                curr_input = tl.load(
                    curr_input_pointer, mask=mask, other=0.0
                ).to(tl.float32)

                sum_acc += curr_input
                sq_sum_acc += curr_input * curr_input

        count = batch_dim * spatial_dim
        channel_sum = tl.sum(tl.sum(sum_acc, axis=0), axis=0)
        channel_sq_sum = tl.sum(tl.sum(sq_sum_acc, axis=0), axis=0)

        mean = channel_sum / count
        var = channel_sq_sum / count - mean * mean
        var = tl.maximum(var, 0.0)
        inv_std = tl.math.rsqrt(var + eps)

        running_mean = tl.load(running_mean_pointer + feat_pid)
        running_var = tl.load(running_var_pointer + feat_pid)
        unbiased_var = tl.where(count > 1, var * count / (count - 1), var)

        tl.store(
            running_mean_pointer + feat_pid,
            (1 - momentum) * running_mean + momentum * mean,
        )
        tl.store(
            running_var_pointer + feat_pid,
            (1 - momentum) * running_var + momentum * unbiased_var,
        )
    else:
        mean = tl.load(running_mean_pointer + feat_pid).to(tl.float32)
        var = tl.load(running_var_pointer + feat_pid).to(tl.float32)
        inv_std = tl.math.rsqrt(var + eps)

    if weight_pointer is None:
        weight = 1.0
    else:
        weight = tl.load(weight_pointer + feat_pid).to(tl.float32)
    if bias_pointer is None:
        bias = 0.0
    else:
        bias = tl.load(bias_pointer + feat_pid).to(tl.float32)

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
        batch_mask = batch_offset < batch_dim

        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim
            mask = batch_mask[:, None] & spatial_mask[None, :]

            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_output_pointer = (
                output_pointer
                + output_feat_stride * feat_pid
                + output_batch_stride * batch_offset[:, None]
                + output_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(curr_input_pointer, mask=mask, other=0.0).to(
                tl.float32
            )
            output = weight * (curr_input - mean) * inv_std + bias
            tl.store(curr_output_pointer, output, mask=mask)


def _validate_batch_norm_inputs(
    x,
    running_mean,
    running_var,
    weight,
    bias,
    training,
    momentum,
    eps,
):
    if not isinstance(x, torch.Tensor):
        raise TypeError("batch_norm expects a torch.Tensor input")
    if x.device.type != "cpu":
        raise ValueError("batch_norm only supports CPU inputs in this example")
    if x.dtype != torch.float32:
        raise ValueError("batch_norm only supports torch.float32 inputs in this example")
    if x.ndim not in (2, 3, 4):
        raise ValueError("batch_norm expects an input shaped like (N, C), (N, C, L), or (N, C, H, W)")
    if any(dim <= 0 for dim in x.shape):
        raise ValueError("batch_norm requires all input dimensions to be non-zero")
    if x.shape[0] <= 0 or x.shape[1] <= 0:
        raise ValueError("batch_norm requires positive N and C dimensions")

    channels = x.shape[1]
    for name, tensor in (("running_mean", running_mean), ("running_var", running_var)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must have dtype torch.float32")
        if tuple(tensor.shape) != (channels,):
            raise ValueError(f"{name} must have shape ({channels},)")

    for name, tensor in (("weight", weight), ("bias", bias)):
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must have dtype torch.float32")
        if tuple(tensor.shape) != (channels,):
            raise ValueError(f"{name} must have shape ({channels},)")

    if not isinstance(training, bool):
        raise TypeError("training must be a bool")
    if isinstance(momentum, bool) or not isinstance(momentum, (int, float)):
        raise TypeError("momentum must be a finite number")
    if not isfinite(float(momentum)):
        raise ValueError("momentum must be finite")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("eps must be a finite positive number")
    if not isfinite(float(eps)):
        raise ValueError("eps must be finite")
    if eps <= 0:
        raise ValueError("eps must be a positive value")

    if training:
        input_3d = make_3d_for_bn(x)
        batch_dim, _, spatial_dim = input_3d.shape
        if batch_dim * spatial_dim <= 1:
            raise ValueError(
                "Expected more than 1 value per channel when training"
            )
        if not running_mean.is_contiguous():
            raise ValueError(
                "running_mean must be contiguous for caller-visible in-place updates"
            )
        if not running_var.is_contiguous():
            raise ValueError(
                "running_var must be contiguous for caller-visible in-place updates"
            )


def triton_batch_norm(
    x,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
):
    _validate_batch_norm_inputs(
        x, running_mean, running_var, weight, bias, training, momentum, eps
    )
    

    input_3d = make_3d_for_bn(x.contiguous())
    weight = None if weight is None else weight.contiguous().reshape(-1)
    bias = None if bias is None else bias.contiguous().reshape(-1)
    if not training:
        running_mean = running_mean.contiguous()
        running_var = running_var.contiguous()

    batch_dim, feat_dim, spatial_dim = input_3d.shape
    output_3d = torch.empty_like(input_3d)

    with _use_cpu_driver():
        batch_norm_forward_kernel[(feat_dim,)](
            input_3d,
            weight,
            bias,
            output_3d,
            running_mean,
            running_var,
            batch_dim,
            spatial_dim,
            *input_3d.stride(),
            *output_3d.stride(),
            float(momentum),
            float(eps),
            is_train=training,
            BLOCK_M=16,
            BLOCK_N=16,
        )

    return output_3d.view_as(x)


def make_batch_norm_inputs(shape, *, with_affine=True):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    channels = shape[1]
    running_mean = torch.randn((channels,), dtype=torch.float32, device="cpu")
    running_var = torch.rand((channels,), dtype=torch.float32, device="cpu") + 0.5
    if with_affine:
        weight = torch.randn((channels,), dtype=torch.float32, device="cpu")
        bias = torch.randn((channels,), dtype=torch.float32, device="cpu")
    else:
        weight = None
        bias = None
    return x, weight, bias, running_mean, running_var


def batch_norm_reference(
    x,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
):
    return F.batch_norm(
        x,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )


def test_triton_batch_norm_matches_torch_training():
    training_cases = [
        ((4, 3), True),
        ((4, 3, 5), False),
        ((2, 4, 3, 3), True),
    ]
    for shape, with_affine in training_cases:
        x, weight, bias, running_mean, running_var = make_batch_norm_inputs(
            shape, with_affine=with_affine
        )
        ref_running_mean = running_mean.clone()
        ref_running_var = running_var.clone()

        ref = batch_norm_reference(
            x,
            ref_running_mean,
            ref_running_var,
            weight=weight,
            bias=bias,
            training=True,
        )
        out = triton_batch_norm(
            x,
            running_mean,
            running_var,
            weight=weight,
            bias=bias,
            training=True,
        )

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(running_mean, ref_running_mean, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(running_var, ref_running_var, atol=1e-4, rtol=1e-4)

    singleton_x, singleton_weight, singleton_bias, singleton_mean, singleton_var = (
        make_batch_norm_inputs((1, 3), with_affine=True)
    )
    with pytest.raises(ValueError, match="Expected more than 1 value per channel when training"):
        triton_batch_norm(
            singleton_x,
            singleton_mean,
            singleton_var,
            weight=singleton_weight,
            bias=singleton_bias,
            training=True,
        )


def test_triton_batch_norm_matches_torch_eval_without_affine():
    x, _, _, running_mean, running_var = make_batch_norm_inputs(
        (2, 4, 3, 3), with_affine=False
    )
    original_running_mean = running_mean.clone()
    original_running_var = running_var.clone()
    out = triton_batch_norm(
        x,
        running_mean,
        running_var,
        training=False,
    )
    ref = batch_norm_reference(
        x,
        running_mean.clone(),
        running_var.clone(),
        training=False,
    )

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(running_mean, original_running_mean, atol=0.0, rtol=0.0)
    torch.testing.assert_close(running_var, original_running_var, atol=0.0, rtol=0.0)

    noncontig_mean = torch.randn((4, 2), dtype=torch.float32, device="cpu")[:, 0]
    noncontig_var = torch.rand((4, 2), dtype=torch.float32, device="cpu")[:, 0]
    assert not noncontig_mean.is_contiguous()
    assert not noncontig_var.is_contiguous()
    original_noncontig_mean = noncontig_mean.clone()
    original_noncontig_var = noncontig_var.clone()
    out_noncontig = triton_batch_norm(
        x,
        noncontig_mean,
        noncontig_var,
        training=False,
    )
    ref_noncontig = batch_norm_reference(
        x,
        noncontig_mean.clone(),
        noncontig_var.clone(),
        training=False,
    )
    torch.testing.assert_close(out_noncontig, ref_noncontig, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        noncontig_mean, original_noncontig_mean, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        noncontig_var, original_noncontig_var, atol=0.0, rtol=0.0
    )
