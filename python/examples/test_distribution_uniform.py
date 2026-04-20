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
_DEFAULT_PHILOX_GENERATOR = torch.Generator(device="cpu")
_DEFAULT_PHILOX_GENERATOR.manual_seed(
    int.from_bytes(os.urandom(8), "little") % (2**63 - 1)
)


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





def _validate_uniform_destination(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.device.type != "cpu":
        raise ValueError("uniform_ only supports CPU tensors in this example")
    if x.dtype != torch.float32:
        raise ValueError("uniform_ only supports torch.float32 tensors in this example")
    if not x.is_contiguous():
        raise ValueError("uniform_ expects a contiguous destination tensor")


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
def _uniform_scale_kernel(raw_ptr, out_ptr, n_elements, from_, to, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    raw = tl.load(raw_ptr + offsets, mask=mask, other=0).to(tl.int32)
    values = _uint_to_uniform_float(raw) * (to - from_) + from_
    tl.store(out_ptr + offsets, values, mask=mask)


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def _uniform_kernel(out_ptr, n_elements, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    zero = c0 * 0

    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, zero, zero)
    y0 = _uint_to_uniform_float(r0) * (to - from_) + from_
    y1 = _uint_to_uniform_float(r1) * (to - from_) + from_
    y2 = _uint_to_uniform_float(r2) * (to - from_) + from_
    y3 = _uint_to_uniform_float(r3) * (to - from_) + from_

    off0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
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


def _launch_uniform(out, from_, to, *, seed=None, offset=None):
    n_elements = out.numel()
    if n_elements == 0:
        return

    if seed is None or offset is None:
        seed, offset = _next_default_philox_seed_offset()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"] * UNROLL),)
    _uniform_kernel[grid](out, n_elements, seed, offset, from_, to, BLOCK=BLOCK)


def run_uniform_scale_transform(raw_values, from_, to):
    if not isinstance(raw_values, torch.Tensor):
        raise TypeError("raw_values must be a torch.Tensor")
    if raw_values.device.type != "cpu":
        raise ValueError("run_uniform_scale_transform only supports CPU tensors")
    if raw_values.dtype != torch.int32:
        raise ValueError("run_uniform_scale_transform expects torch.int32 tensors")
    if not raw_values.is_contiguous():
        raise ValueError("run_uniform_scale_transform expects contiguous inputs")
    if not to > from_:
        raise ValueError("to must be greater than from_")

    
    out = torch.empty(raw_values.shape, dtype=torch.float32, device="cpu")
    n_elements = raw_values.numel()

    if n_elements == 0:
        return out

    grid = (triton.cdiv(n_elements, BLOCK),)
    with _use_cpu_driver():
        _uniform_scale_kernel[grid](raw_values, out, n_elements, from_, to, BLOCK=BLOCK)
    return out


def uniform_(x, from_=0.0, to=1.0):
    _validate_uniform_destination(x)
    if not to > from_:
        raise ValueError("to must be greater than from_")

    
    with _use_cpu_driver():
        _launch_uniform(x, from_, to)
    return x


def test_uniform_scale_transform_matches_formula():
    raw_values = torch.tensor([1, 7, 31, 1024], dtype=torch.int32, device="cpu")
    out = run_uniform_scale_transform(raw_values, -1.5, 2.0)

    ref = raw_values.to(torch.float32) * 4.6566127342e-10
    ref = ref * (2.0 - (-1.5)) + (-1.5)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=0)


def test_uniform_returns_same_tensor_and_respects_bounds():
    x = torch.empty((4096,), dtype=torch.float32, device="cpu")
    out = uniform_(x, from_=-2.0, to=3.0)

    assert out is x
    assert out.dtype == torch.float32
    assert out.shape == (4096,)
    assert torch.all(out >= -2.0)
    assert torch.all(out < 3.0)


def test_uniform_statistics_roughly_match_pytorch():
    x = torch.empty((8192,), dtype=torch.float32, device="cpu")
    ref = torch.empty_like(x)

    uniform_(x, from_=-1.0, to=2.0)
    torch.manual_seed(0)
    ref.uniform_(-1.0, 2.0)

    expected_mean = 0.5
    expected_std = (2.0 - (-1.0)) / (12.0**0.5)
    assert abs(x.mean().item() - expected_mean) < 0.08
    assert abs(x.std(unbiased=False).item() - expected_std) < 0.08
    assert abs(x.mean().item() - ref.mean().item()) < 0.08
    assert abs(x.std(unbiased=False).item() - ref.std(unbiased=False).item()) < 0.08


def test_uniform_rejects_invalid_range():
    x = torch.empty((32,), dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="to must be greater than from_"):
        uniform_(x, from_=1.0, to=1.0)


def test_uniform_successive_default_calls_differ():
    x1 = torch.empty((1024,), dtype=torch.float32, device="cpu")
    x2 = torch.empty_like(x1)

    uniform_(x1)
    uniform_(x2)

    assert not torch.equal(x1, x2)
