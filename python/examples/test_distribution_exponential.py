import os
from contextlib import contextmanager
from tempfile import mkdtemp

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver

_BLOCK = 256
_UNROLL = 4
_DEFAULT_PHILOX_GENERATOR = torch.Generator(device="cpu")
_DEFAULT_PHILOX_GENERATOR.manual_seed(int.from_bytes(os.urandom(8), "little") % (2**63 - 1))


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





def _validate_exponential_destination(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.device.type != "cpu":
        raise ValueError("exponential_ only supports CPU tensors in this example")
    if x.dtype != torch.float32:
        raise ValueError("exponential_ only supports torch.float32 tensors in this example")
    if not x.is_contiguous():
        raise ValueError("exponential_ expects a contiguous destination tensor")


def _validate_exponential_rate(lambd):
    if lambd <= 0:
        raise ValueError("lambd must be positive")


def _validate_exponential_transform_input(u):
    _validate_exponential_destination(u)
    if not torch.isfinite(u).all():
        raise ValueError("u must contain only finite values")
    if torch.any(u < 0) or torch.any(u >= 1):
        raise ValueError("u must contain values in [0, 1)")


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
def _safe_fast_log_f32(x):
    min_normal = (x * 0.0 + 1.17549435e-38).to(tl.float32)
    max_u = x * 0.0 + 0.99999994
    x = tl.minimum(tl.maximum(x, min_normal), max_u)
    bits = x.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) - 127
    mantissa = (bits & 0x7FFFFF).to(tl.float32) * (1.0 / 8388608.0) + 1.0
    m1 = mantissa - 1.0
    return (
        m1 * (1.0 + m1 * (-0.5 + m1 * (0.3333333333 - m1 * 0.25)))
        + exponent.to(tl.float32) * 0.6931471805599453
    )


@triton.jit
def _transform_exponential_f32_fast(u, inv_lambd, eps_minus):
    log_u = tl.where(u >= 1.0 + eps_minus, eps_minus, _safe_fast_log_f32(u))
    return -inv_lambd * log_u


@triton.jit
def _exponential_transform_kernel(u_ptr, out_ptr, n_elements, inv_lambd, eps_minus, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    u = tl.load(u_ptr + offsets, mask=mask, other=0.5).to(tl.float32)
    values = _transform_exponential_f32_fast(u, inv_lambd, eps_minus)
    tl.store(out_ptr + offsets, values, mask=mask)


@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "n_elements"])
def _exponential_kernel_f32(out_ptr, n_elements, inv_lambd, eps_minus, philox_seed, philox_offset, BLOCK: tl.constexpr):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    c0 += i
    zero = c0 * 0

    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, zero, zero)
    y0 = _transform_exponential_f32_fast(_uint_to_uniform_float(r0), inv_lambd, eps_minus)
    y1 = _transform_exponential_f32_fast(_uint_to_uniform_float(r1), inv_lambd, eps_minus)
    y2 = _transform_exponential_f32_fast(_uint_to_uniform_float(r2), inv_lambd, eps_minus)
    y3 = _transform_exponential_f32_fast(_uint_to_uniform_float(r3), inv_lambd, eps_minus)

    off0 = pid.to(tl.uint64) * BLOCK * 4 + tl.arange(0, BLOCK)
    off1 = off0 + BLOCK
    off2 = off1 + BLOCK
    off3 = off2 + BLOCK

    tl.store(out_ptr + off0, y0, mask=off0 < n_elements)
    tl.store(out_ptr + off1, y1, mask=off1 < n_elements)
    tl.store(out_ptr + off2, y2, mask=off2 < n_elements)
    tl.store(out_ptr + off3, y3, mask=off3 < n_elements)


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


def _launch_exponential_transform(u, lambd):
    n_elements = u.numel()
    if n_elements == 0:
        return torch.empty_like(u)

    out = torch.empty_like(u)
    eps_minus = -0.5 * torch.finfo(torch.float32).eps
    grid = (triton.cdiv(n_elements, _BLOCK),)
    with _use_cpu_driver():
        _exponential_transform_kernel[grid](
            u,
            out,
            n_elements,
            1.0 / lambd,
            eps_minus,
            BLOCK=_BLOCK,
        )
    return out


def _launch_exponential(out, lambd, *, seed=None, offset=None):
    n_elements = out.numel()
    if n_elements == 0:
        return

    if seed is None or offset is None:
        seed, offset = _next_default_philox_seed_offset()

    eps_minus = -0.5 * torch.finfo(torch.float32).eps
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"] * _UNROLL),)
    with _use_cpu_driver():
        _exponential_kernel_f32[grid](
            out,
            n_elements,
            1.0 / lambd,
            eps_minus,
            seed,
            offset,
            BLOCK=_BLOCK,
        )


def run_exponential_transform(u, lambd):
    _validate_exponential_transform_input(u)
    _validate_exponential_rate(lambd)
    
    return _launch_exponential_transform(u, lambd)


def exponential_(x, lambd=1.0):
    _validate_exponential_destination(x)
    _validate_exponential_rate(lambd)

    
    _launch_exponential(x, lambd)
    return x


def test_exponential_transform_matches_torch_formula():
    u = torch.tensor([0.125, 0.25, 0.5, 0.75], dtype=torch.float32, device="cpu")
    out = run_exponential_transform(u, lambd=1.7)
    ref = -torch.log(u) / 1.7

    torch.testing.assert_close(out, ref, atol=3e-3, rtol=0)


def test_exponential_transform_boundary_behavior_is_finite_and_clamped():
    min_normal = torch.tensor(torch.finfo(torch.float32).tiny, dtype=torch.float32)
    almost_one = torch.nextafter(
        torch.tensor(1.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)
    )
    u = torch.tensor(
        [0.0, min_normal.item(), almost_one.item()],
        dtype=torch.float32,
        device="cpu",
    )
    out = run_exponential_transform(u, lambd=1.7)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[0], out[1], atol=0.0, rtol=0.0)
    assert out[0] > 0.0
    assert out[2] > 0.0
    assert out[2] < out[0]


def test_exponential_returns_same_tensor_and_is_nonnegative():
    x = torch.empty((4096,), dtype=torch.float32, device="cpu")
    out = exponential_(x, lambd=2.0)

    assert out is x
    assert out.dtype == torch.float32
    assert out.shape == (4096,)
    assert torch.all(out >= 0.0)


def test_exponential_statistics_roughly_match_pytorch():
    x = torch.empty((8192,), dtype=torch.float32, device="cpu")
    ref = torch.empty_like(x)

    exponential_(x, lambd=1.75)
    torch.manual_seed(0)
    ref.exponential_(1.75)

    expected_mean = 1.0 / 1.75
    expected_std = 1.0 / 1.75
    assert abs(x.mean().item() - expected_mean) < 0.08
    assert abs(x.std(unbiased=False).item() - expected_std) < 0.08
    assert abs(x.mean().item() - ref.mean().item()) < 0.10
    assert abs(x.std(unbiased=False).item() - ref.std(unbiased=False).item()) < 0.10


def test_exponential_successive_default_calls_differ():
    x1 = torch.empty((1024,), dtype=torch.float32, device="cpu")
    x2 = torch.empty_like(x1)

    exponential_(x1)
    exponential_(x2)

    assert not torch.equal(x1, x2)


def test_exponential_rejects_invalid_rate():
    x = torch.empty((16,), dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="lambd must be positive"):
        exponential_(x, lambd=0.0)


def test_exponential_transform_rejects_invalid_rate():
    u = torch.tensor([0.25, 0.5], dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="lambd must be positive"):
        run_exponential_transform(u, lambd=0.0)


def test_exponential_transform_rejects_one():
    u = torch.tensor([0.25, 1.0], dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="u must contain values in \\[0, 1\\)"):
        run_exponential_transform(u, lambd=1.0)
