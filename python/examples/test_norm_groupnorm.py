import os
from pathlib import Path
from tempfile import mkdtemp
from contextlib import contextmanager

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




@triton.jit
def group_norm_kernel(
    X,
    Y,
    W,
    B,
    group_size,
    C,
    HW,
    num_groups,
    eps,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW

    group_offsets = tl.arange(0, BLOCK_GROUP_SIZE)
    hw_offsets = tl.arange(0, BLOCK_HW_SIZE)

    wb_offsets = group * group_size + group_offsets
    wb_mask = wb_offsets < C

    xy_offsets = pid * num_elements + group_offsets[:, None] * HW + hw_offsets[None, :]
    xy_mask = wb_mask[:, None] & (hw_offsets[None, :] < HW)

    x = tl.load(X + xy_offsets, mask=xy_mask, other=0.0).to(tl.float32)
    mean = tl.sum(x) / num_elements
    centered = tl.where(xy_mask, x - mean, 0.0)

    var = tl.sum(centered * centered) / num_elements
    rstd = tl.math.rsqrt(var + eps)

    if W is None:
        weight = 1.0
    else:
        weight = tl.load(W + wb_offsets, mask=wb_mask, other=0.0)[:, None]

    if B is None:
        bias = 0.0
    else:
        bias = tl.load(B + wb_offsets, mask=wb_mask, other=0.0)[:, None]

    y = centered * rstd * weight + bias

    tl.store(Y + xy_offsets, y, mask=xy_mask)


def _validate_group_norm_inputs(x, num_groups, weight, bias):
    if not isinstance(x, torch.Tensor):
        raise TypeError("group_norm expects a torch.Tensor input")
    if x.device.type != "cpu":
        raise ValueError("group_norm only supports CPU inputs in this example")
    if x.dtype != torch.float32:
        raise ValueError("group_norm only supports torch.float32 inputs in this example")
    if x.ndim != 4:
        raise ValueError("group_norm expects a 4D tensor shaped as (N, C, H, W)")
    if not isinstance(num_groups, int) or num_groups <= 0:
        raise ValueError("num_groups must be a positive integer")
    if any(dim <= 0 for dim in x.shape):
        raise ValueError("group_norm requires all input dimensions to be non-zero")

    _, channels, _, _ = x.shape
    if channels % num_groups != 0:
        raise ValueError("group_norm requires C % num_groups == 0")

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


def triton_group_norm(x, num_groups, weight=None, bias=None, eps=1e-5):
    _validate_group_norm_inputs(x, num_groups, weight, bias)

    x = x.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    n, c, h, w = x.shape
    hw = h * w
    group_size = c // num_groups

    y = torch.empty_like(x)
    with _use_cpu_driver():
        group_norm_kernel[(n * num_groups,)](
            x,
            y,
            weight,
            bias,
            group_size,
            c,
            hw,
            num_groups,
            eps,
            BLOCK_GROUP_SIZE=triton.next_power_of_2(group_size),
            BLOCK_HW_SIZE=triton.next_power_of_2(hw),
        )
    return y


def make_group_norm_inputs(shape=(2, 8, 4, 4), *, with_affine=True):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    if with_affine:
        weight = torch.randn((shape[1],), dtype=torch.float32, device="cpu")
        bias = torch.randn((shape[1],), dtype=torch.float32, device="cpu")
    else:
        weight = None
        bias = None
    return x, weight, bias


def group_norm_reference(x, num_groups, weight=None, bias=None, eps=1e-5):
    return F.group_norm(x, num_groups, weight=weight, bias=bias, eps=eps)


def test_triton_groupnorm_matches_torch_basic():
    x, weight, bias = make_group_norm_inputs(shape=(2, 8, 4, 4), with_affine=True)
    out = triton_group_norm(x, 4, weight=weight, bias=bias)
    ref = group_norm_reference(x, 4, weight=weight, bias=bias)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_triton_groupnorm_matches_torch_without_affine():
    x, _, _ = make_group_norm_inputs(shape=(1, 6, 3, 5), with_affine=False)
    out = triton_group_norm(x, 3)
    ref = group_norm_reference(x, 3)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
