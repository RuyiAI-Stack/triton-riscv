import os
from math import isfinite
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp

import pytest
import torch
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
            # Restore the cold-start state without forcing Triton to discover
            # a default device driver that may not exist in this example.
            triton.runtime.driver._active = None
        else:
            triton.runtime.driver.set_active(previous_driver)





def _as_tuple(normalized_shape):
    if isinstance(normalized_shape, int):
        return (normalized_shape,)
    return tuple(normalized_shape)


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@triton.jit(do_not_specialize=["eps"])
def layer_norm_persistent_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(0)

    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N

    x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x) / N
    centered = x - mean
    sum_square = tl.sum(tl.where(mask, centered * centered, 0.0))
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    if weight_ptr is None:
        weight = 1.0
    else:
        weight = tl.load(weight_ptr + n_offsets, mask=mask, other=0.0)
    if bias_ptr is None:
        bias = 0.0
    else:
        bias = tl.load(bias_ptr + n_offsets, mask=mask, other=0.0)

    out = centered * rstd * weight + bias
    tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)


@triton.jit(do_not_specialize=["eps"])
def layer_norm_persistent_kernel_multiline(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    M,
    N,
    eps,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_offsets < M

    n_offsets = tl.arange(0, TILE_N)[None, :]
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask

    x = tl.load(in_ptr + m_offsets[:, None] * N + n_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    mean = tl.sum(x, axis=1) / N
    centered = x - mean[:, None]
    sum_square = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    if weight_ptr is None:
        weight = 1.0
    else:
        weight = tl.load(weight_ptr + n_offsets, mask=n_mask, other=0.0)
    if bias_ptr is None:
        bias = 0.0
    else:
        bias = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)

    out = centered * rstd[:, None] * weight + bias
    tl.store(out_ptr + m_offsets[:, None] * N + n_offsets, out, mask=mask)


@triton.jit(do_not_specialize=["eps"])
def layer_norm_loop_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(0)

    mean_tiles = tl.zeros((TILE_N,), dtype=tl.float32)
    sum_tiles = tl.zeros((TILE_N,), dtype=tl.float32)
    counts = tl.zeros((TILE_N,), dtype=tl.int32)
    num_steps = tl.cdiv(N, TILE_N)

    for step in range(0, num_steps - 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        new_mean = mean_tiles + (x - mean_tiles) / (step + 1)
        new_sum = sum_tiles + (x - new_mean) * (x - mean_tiles)
        counts += 1
        mean_tiles = new_mean
        sum_tiles = new_sum

    for step in range(num_steps - 1, num_steps):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
        new_mean = tl.where(mask, mean_tiles + (x - mean_tiles) / (step + 1), mean_tiles)
        new_sum = tl.where(mask, sum_tiles + (x - new_mean) * (x - mean_tiles), sum_tiles)
        counts += mask.to(tl.int32)
        mean_tiles = new_mean
        sum_tiles = new_sum

    final_mean = tl.sum(mean_tiles * counts) / N
    var = (
        tl.sum(sum_tiles + counts * (mean_tiles - final_mean) * (mean_tiles - final_mean))
        / N
    )
    rstd = tl.math.rsqrt(var + eps)

    prev_multiple = prev_multiple_of(N, TILE_N)

    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        if weight_ptr is None:
            weight = 1.0
        else:
            weight = tl.load(weight_ptr + n_offsets, mask=mask, other=0.0)
        if bias_ptr is None:
            bias = 0.0
        else:
            bias = tl.load(bias_ptr + n_offsets, mask=mask, other=0.0)
        out = weight * (x - final_mean) * rstd + bias
        tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets, eviction_policy="evict_first").to(
            tl.float32
        )
        if weight_ptr is None:
            weight = 1.0
        else:
            weight = tl.load(weight_ptr + n_offsets)
        if bias_ptr is None:
            bias = 0.0
        else:
            bias = tl.load(bias_ptr + n_offsets)
        out = weight * (x - final_mean) * rstd + bias
        tl.store(out_ptr + pid * N + n_offsets, out)


def _validate_layer_norm_inputs(x, normalized_shape, weight, bias, eps):
    if not isinstance(x, torch.Tensor):
        raise TypeError("layer_norm expects a torch.Tensor input")
    if x.device.type != "cpu":
        raise ValueError("layer_norm only supports CPU inputs in this example")
    if x.dtype != torch.float32:
        raise ValueError("layer_norm only supports torch.float32 inputs in this example")
    if x.ndim == 0:
        raise ValueError("layer_norm expects an input tensor with at least one dimension")

    normalized_shape = _as_tuple(normalized_shape)
    if not normalized_shape:
        raise ValueError("normalized_shape must contain at least one dimension")
    if any(not isinstance(dim, int) or dim <= 0 for dim in normalized_shape):
        raise ValueError("normalized_shape must contain positive integers")
    if len(normalized_shape) > x.ndim:
        raise ValueError("normalized_shape must not have more dimensions than the input")
    if tuple(x.shape[-len(normalized_shape) :]) != normalized_shape:
        raise ValueError("input trailing dimensions must match normalized_shape")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("eps must be a finite positive number")
    if not isfinite(float(eps)):
        raise ValueError("eps must be finite")
    if eps <= 0:
        raise ValueError("eps must be a positive value")

    for name, tensor in (("weight", weight), ("bias", bias)):
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must have dtype torch.float32")
        if tuple(tensor.shape) != normalized_shape:
            raise ValueError(f"{name} must have shape {normalized_shape}")

    return normalized_shape


def triton_layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    normalized_shape = _validate_layer_norm_inputs(
        x, normalized_shape, weight, bias, eps
    )

    x = x.contiguous()
    weight = None if weight is None else weight.contiguous().reshape(-1)
    bias = None if bias is None else bias.contiguous().reshape(-1)

    N = 1
    for dim in normalized_shape:
        N *= dim
    M = x.numel() // N

    x_2d = x.reshape(M, N)
    y_2d = torch.empty_like(x_2d)

    with _use_cpu_driver():
        if N <= 128:
            tile_n = triton.next_power_of_2(N)
            tile_m = triton.cdiv(1024, tile_n)
            grid = (triton.cdiv(M, tile_m),)
            layer_norm_persistent_kernel_multiline[grid](
                x_2d,
                y_2d,
                weight,
                bias,
                M,
                N,
                eps,
                TILE_M=tile_m,
                TILE_N=tile_n,
            )
        elif N <= 4096:
            tile_n = triton.next_power_of_2(N)
            grid = (M,)
            layer_norm_persistent_kernel[grid](
                x_2d,
                y_2d,
                weight,
                bias,
                M,
                N,
                eps,
                TILE_N=tile_n,
            )
        else:
            grid = (M,)
            layer_norm_loop_kernel[grid](
                x_2d,
                y_2d,
                weight,
                bias,
                M,
                N,
                eps,
                TILE_N=1024,
            )

    return y_2d.reshape(x.shape)


def make_layer_norm_inputs(shape, *, with_affine=True):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    normalized_shape = shape[1:]
    if with_affine:
        weight = torch.randn(normalized_shape, dtype=torch.float32, device="cpu")
        bias = torch.randn(normalized_shape, dtype=torch.float32, device="cpu")
    else:
        weight = None
        bias = None
    return x, normalized_shape, weight, bias


def layer_norm_reference(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    return torch.layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=eps)


def _assert_layer_norm_matches_reference(shape, *, with_affine, eps=1e-5):
    x, normalized_shape, weight, bias = make_layer_norm_inputs(
        shape, with_affine=with_affine
    )
    out = triton_layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=eps)
    ref = layer_norm_reference(x, normalized_shape, weight=weight, bias=bias, eps=eps)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_triton_layernorm_matches_torch_small_n():
    _assert_layer_norm_matches_reference((17, 128), with_affine=True)
    _assert_layer_norm_matches_reference((5, 64), with_affine=False)
    _assert_layer_norm_matches_reference((9, 4, 32), with_affine=True)
    x, normalized_shape, weight, bias = make_layer_norm_inputs(
        (3, 128), with_affine=True
    )
    with pytest.raises(ValueError, match="eps must be a positive value"):
        triton_layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=0.0)


def test_triton_layernorm_matches_torch_medium_n():
    _assert_layer_norm_matches_reference((31, 129), with_affine=True)
    _assert_layer_norm_matches_reference((7, 513), with_affine=False)
    _assert_layer_norm_matches_reference((13, 4096), with_affine=True)
    _assert_layer_norm_matches_reference((33, 16, 16), with_affine=True)
    x, normalized_shape, weight, bias = make_layer_norm_inputs(
        (4, 129), with_affine=True
    )
    with pytest.raises(ValueError, match="eps must be a positive value"):
        triton_layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=-1e-5)
    with pytest.raises(TypeError, match="eps must be a finite positive number"):
        triton_layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=None)


def test_triton_layernorm_matches_torch_large_n_without_affine():
    _assert_layer_norm_matches_reference((2, 4097), with_affine=False)
    _assert_layer_norm_matches_reference((1, 40999), with_affine=False)
    _assert_layer_norm_matches_reference((2, 4097), with_affine=True)
    x, normalized_shape, weight, bias = make_layer_norm_inputs(
        (2, 4097), with_affine=True
    )
    with pytest.raises(TypeError, match="eps must be a finite positive number"):
        triton_layer_norm(x, normalized_shape, weight=weight, bias=bias, eps="1e-5")
    with pytest.raises(ValueError, match="eps must be finite"):
        triton_layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=float("nan"))
    with pytest.raises(ValueError, match="eps must be finite"):
        triton_layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=float("inf"))
