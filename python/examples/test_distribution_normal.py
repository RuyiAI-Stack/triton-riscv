import os
from contextlib import contextmanager
from tempfile import mkdtemp

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver

BLOCK = 256
UNROLL = 4
DEFAULT_PHILOX_SEED = 0x1234ABCD
DEFAULT_PHILOX_OFFSET = 0
_DEFAULT_PHILOX_GENERATOR = torch.Generator(device="cpu")
_DEFAULT_PHILOX_GENERATOR.manual_seed(DEFAULT_PHILOX_SEED ^ DEFAULT_PHILOX_OFFSET)





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
            triton.runtime.driver._active = None
        else:
            triton.runtime.driver.set_active(previous_driver)


def _validate_normal_destination(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.device.type != "cpu":
        raise ValueError("normal_ only supports CPU tensors in this example")
    if x.dtype != torch.float32:
        raise ValueError("normal_ only supports torch.float32 tensors in this example")
    if not x.is_contiguous():
        raise ValueError("normal_ expects a contiguous destination tensor")


def _validate_box_muller_inputs(u1, u2):
    if not isinstance(u1, torch.Tensor) or not isinstance(u2, torch.Tensor):
        raise TypeError("u1 and u2 must be torch.Tensor instances")
    if u1.device.type != "cpu" or u2.device.type != "cpu":
        raise ValueError("run_box_muller_transform only supports CPU tensors")
    if u1.dtype != torch.float32 or u2.dtype != torch.float32:
        raise ValueError("run_box_muller_transform expects torch.float32 tensors")
    if u1.shape != u2.shape:
        raise ValueError("u1 and u2 must have the same shape")
    if not u1.is_contiguous() or not u2.is_contiguous():
        raise ValueError("u1 and u2 must be contiguous")
    if not torch.isfinite(u1).all() or torch.any((u1 <= 0) | (u1 > 1.0)):
        raise ValueError("u1 must contain only finite values in (0, 1]")
    if not torch.isfinite(u2).all():
        raise ValueError("u2 must contain only finite values")


@triton.jit
def _uint_to_uniform_float(x):
    if tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int32):
        x = x.to(tl.int32, bitcast=True)
        scale = 4.6566127342e-10
    else:
        tl.static_assert(
            tl.constexpr(x.dtype == tl.uint64) or tl.constexpr(x.dtype == tl.int64)
        )
        x = x.to(tl.int64, bitcast=True)
        scale = 1.0842020432385337e-19
    x = tl.where(x < 0, -x - 1, x)
    return x * scale


@triton.jit
def _high_precision_fast_sin_cos(x):
    two_pi = 6.283185307179586
    x = x - two_pi * tl.floor(x / two_pi + 0.5)
    x2 = x * x

    s_c0 = 0.99999999999999999999
    s_c1 = -0.16666666666666666654
    s_c2 = 0.00833333333333332876
    s_c3 = -0.00019841269841269616
    s_c4 = 2.755731922398589e-6
    s_c5 = -2.505210838544172e-8

    sin_x = x * (
        s_c0 + x2 * (s_c1 + x2 * (s_c2 + x2 * (s_c3 + x2 * (s_c4 + x2 * s_c5))))
    )

    c_c0 = 1.0
    c_c1 = -0.49999999999999999983
    c_c2 = 0.04166666666666666636
    c_c3 = -0.00138888888888888742
    c_c4 = 2.4801587301587299e-5
    c_c5 = -2.755731922398581e-7

    cos_x = c_c0 + x2 * (c_c1 + x2 * (c_c2 + x2 * (c_c3 + x2 * (c_c4 + x2 * c_c5))))
    return sin_x, cos_x


@triton.jit
def _pair_uniform_to_normal_fast(u1, u2):
    u1 = tl.maximum(1.0e-7, u1)
    theta = 6.283185307179586 * u2
    radius = tl.sqrt(-2.0 * tl.log(u1))
    sin_theta, cos_theta = _high_precision_fast_sin_cos(theta)
    return radius * cos_theta, radius * sin_theta


@triton.jit
def _box_muller_kernel(u1_ptr, u2_ptr, out0_ptr, out1_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    u1 = tl.load(u1_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    u2 = tl.load(u2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out0, out1 = _pair_uniform_to_normal_fast(u1, u2)

    tl.store(out0_ptr + offsets, out0, mask=mask)
    tl.store(out1_ptr + offsets, out1, mask=mask)


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def _randn_kernel(out_ptr, n_elements, philox_seed, philox_offset, BLOCK: tl.constexpr):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    zero = c0 * 0

    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, zero, zero)
    u0 = _uint_to_uniform_float(r0)
    u1 = _uint_to_uniform_float(r1)
    u2 = _uint_to_uniform_float(r2)
    u3 = _uint_to_uniform_float(r3)

    n0, n1 = _pair_uniform_to_normal_fast(u0, u1)
    n2, n3 = _pair_uniform_to_normal_fast(u2, u3)

    off0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off1 = off0 + BLOCK
    off2 = off1 + BLOCK
    off3 = off2 + BLOCK

    tl.store(out_ptr + off0, n0, mask=off0 < n_elements)
    tl.store(out_ptr + off1, n1, mask=off1 < n_elements)
    tl.store(out_ptr + off2, n2, mask=off2 < n_elements)
    tl.store(out_ptr + off3, n3, mask=off3 < n_elements)


@triton.jit
def _affine_kernel(x_ptr, n_elements, mean, std, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    values = values * std + mean
    tl.store(x_ptr + offsets, values, mask=mask)


def run_box_muller_transform(u1, u2):
    _validate_box_muller_inputs(u1, u2)
    out0 = torch.empty_like(u1)
    out1 = torch.empty_like(u2)
    n_elements = u1.numel()

    if n_elements == 0:
        return out0, out1

    grid = (triton.cdiv(n_elements, BLOCK),)
    with _use_cpu_driver():
        _box_muller_kernel[grid](u1, u2, out0, out1, n_elements, BLOCK=BLOCK)
    return out0, out1


def _next_default_philox_seed_offset():
    seed = torch.randint(
        0,
        2**63 - 1,
        (),
        generator=_DEFAULT_PHILOX_GENERATOR,
        device="cpu",
        dtype=torch.int64,
    ).item()
    offset = torch.randint(
        0,
        2**63 - 1,
        (),
        generator=_DEFAULT_PHILOX_GENERATOR,
        device="cpu",
        dtype=torch.int64,
    ).item()
    return seed, offset


def _launch_randn(out, *, seed=None, offset=None):
    n_elements = out.numel()
    if n_elements == 0:
        return

    if seed is None or offset is None:
        seed, offset = _next_default_philox_seed_offset()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"] * UNROLL),)
    _randn_kernel[grid](out, n_elements, seed, offset, BLOCK=BLOCK)


def normal_(x, mean=0.0, std=1.0):
    _validate_normal_destination(x)
    if std < 0:
        raise ValueError("std must be non-negative")


    if x.numel() == 0:
        return x

    with _use_cpu_driver():
        _launch_randn(x)
        grid = (triton.cdiv(x.numel(), BLOCK),)
        _affine_kernel[grid](x, x.numel(), mean, std, BLOCK=BLOCK)
    return x


def test_box_muller_transform_matches_torch_formula():
    u1 = torch.tensor([0.125, 0.25, 0.5, 0.75], dtype=torch.float32, device="cpu")
    u2 = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32, device="cpu")
    out0, out1 = run_box_muller_transform(u1, u2)

    radius = torch.sqrt(-2.0 * torch.log(torch.clamp(u1, min=1.0e-7)))
    theta = 2.0 * torch.pi * u2
    ref0 = radius * torch.cos(theta)
    ref1 = radius * torch.sin(theta)

    torch.testing.assert_close(out0, ref0, atol=2e-3, rtol=0)
    torch.testing.assert_close(out1, ref1, atol=2e-3, rtol=0)


def test_box_muller_transform_rejects_invalid_domain_inputs():
    valid = torch.tensor([0.25, 0.5], dtype=torch.float32, device="cpu")

    with pytest.raises(ValueError, match=r"u1 must contain only finite values in \(0, 1\]"):
        run_box_muller_transform(
            torch.tensor([0.0, 0.5], dtype=torch.float32, device="cpu"),
            valid,
        )

    with pytest.raises(ValueError, match=r"u1 must contain only finite values in \(0, 1\]"):
        run_box_muller_transform(
            torch.tensor([1.25, 0.5], dtype=torch.float32, device="cpu"),
            valid,
        )

    with pytest.raises(ValueError, match="u2 must contain only finite values"):
        run_box_muller_transform(
            valid,
            torch.tensor([0.25, float("nan")], dtype=torch.float32, device="cpu"),
        )


def test_normal_returns_same_tensor_and_fills_in_place():
    x = torch.empty((1024,), dtype=torch.float32, device="cpu")
    out = normal_(x, mean=1.5, std=0.75)
    assert out is x
    assert out.shape == (1024,)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_normal_statistics_roughly_match_pytorch():
    x = torch.empty((8192,), dtype=torch.float32, device="cpu")
    ref = torch.empty_like(x)

    normal_(x, mean=1.25, std=0.5)
    torch.manual_seed(0)
    ref.normal_(mean=1.25, std=0.5)

    assert abs(x.mean().item() - 1.25) < 0.08
    assert abs(x.std(unbiased=False).item() - 0.5) < 0.08
    assert abs(x.mean().item() - ref.mean().item()) < 0.10
    assert abs(x.std(unbiased=False).item() - ref.std(unbiased=False).item()) < 0.10


def test_normal_restores_previous_driver_state():
    previous_driver = getattr(triton.runtime.driver, "_active", None)
    triton.runtime.driver._active = None

    try:
        x = torch.empty((128,), dtype=torch.float32, device="cpu")
        normal_(x)
        assert getattr(triton.runtime.driver, "_active", None) is None
    finally:
        if previous_driver is None:
            triton.runtime.driver._active = None
        else:
            triton.runtime.driver.set_active(previous_driver)


def test_normal_successive_default_calls_differ():
    x1 = torch.empty((1024,), dtype=torch.float32, device="cpu")
    x2 = torch.empty_like(x1)

    normal_(x1, mean=0.0, std=1.0)
    normal_(x2, mean=0.0, std=1.0)

    assert not torch.equal(x1, x2)


def test_normal_rejects_invalid_std():
    x = torch.empty((16,), dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="std must be non-negative"):
        normal_(x, mean=0.0, std=-1.0)
