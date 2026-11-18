import math
import struct

import pytest
import torch

from .frexp import frexp


def _assert_frexp_close(x):
    # frexp decomposes x = mantissa * 2**exponent with |mantissa| in
    # [0.5, 1).  The bit-splice decomposition is exact for every finite
    # fp16/bf16/fp32 value (and preserves 0/inf/NaN payloads), so the
    # comparison against torch.frexp is bit-exact.
    mantissa, exponent = frexp(x)
    ref_m, ref_e = torch.frexp(x)

    assert isinstance(mantissa, torch.Tensor) and isinstance(exponent, torch.Tensor)
    assert mantissa.dtype == x.dtype, "mantissa keeps the input dtype"
    assert exponent.dtype == torch.int32, "exponent is always int32"
    assert mantissa.shape == x.shape and exponent.shape == x.shape

    torch.testing.assert_close(mantissa, ref_m, rtol=0.0, atol=0.0, equal_nan=True)
    torch.testing.assert_close(exponent, ref_e, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_frexp(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * torch.pow(
        2.0, torch.randint(-40, 40, (size,)).float()
    )
    _assert_frexp_close(x)


def test_frexp_special_values():
    x = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan")],
        dtype=torch.float32,
    )
    mantissa, exponent = frexp(x)

    assert mantissa[0].item() == 0.0
    assert math.copysign(1.0, mantissa[0].item()) == 1.0, "+0.0 keeps its sign"
    assert mantissa[1].item() == 0.0
    assert math.copysign(1.0, mantissa[1].item()) == -1.0, "-0.0 keeps its sign"
    assert mantissa[2].item() == float("inf")
    assert mantissa[3].item() == float("-inf")
    assert torch.isnan(mantissa[4])
    assert torch.equal(exponent, torch.zeros(5, dtype=torch.int32))

    _assert_frexp_close(x)


def test_frexp_denormals_and_power_of_two_edges():
    min_subnormal = struct.unpack("f", struct.pack("I", 1))[0]  # 2**-149
    one_ulp_below_1_over_16 = struct.unpack("f", struct.pack("I", 0x3D7FFFFF))[0]
    just_above_one = torch.nextafter(
        torch.tensor(1.0), torch.tensor(2.0)
    ).item()
    x = torch.tensor(
        [
            min_subnormal,  # smallest fp32 subnormal
            min_subnormal * 6,  # 1.5 * 2**-148, still subnormal
            2.0**-126,  # smallest normal
            -min_subnormal,
            one_ulp_below_1_over_16,  # log2-based decomposition traps this
            2.0**-4,
            0.5,
            1.0,
            just_above_one,
            2.0,
            65536.0,
            3.0e38,
        ],
        dtype=torch.float32,
    )
    _assert_frexp_close(x)


def test_frexp_return_order_reconstructs_input():
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32) * torch.pow(
        2.0, torch.randint(-100, 100, (1024,)).float()
    )
    mantissa, exponent = frexp(x)
    # (mantissa, exponent) order: mantissa * 2**exponent rebuilds x exactly
    # (scaling by a power of two is exact in fp32).
    rebuilt = mantissa * torch.pow(2.0, exponent.to(torch.float32))
    torch.testing.assert_close(rebuilt, x, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_frexp_half_dtypes(dtype):
    torch.manual_seed(0)
    x = (
        torch.randn(512, dtype=torch.float32)
        * torch.pow(2.0, torch.randint(-12, 12, (512,)).float())
    ).to(dtype)
    _assert_frexp_close(x)

    specials = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan")], dtype=dtype
    )
    _assert_frexp_close(specials)

    # Subnormals of the storage dtype normalize exactly as well.
    subnormals = torch.tensor(
        [2.0**-24, 2.0**-16, 6.0e-8], dtype=torch.float32
    ).to(dtype)
    _assert_frexp_close(subnormals)


def test_frexp_rejects_float64():
    # Upstream mirror: the fp32-internal decomposition cannot preserve fp64
    # precision/range, so float64 is rejected explicitly.
    x = torch.randn(8, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="does not support float64"):
        frexp(x)


def test_frexp_rejects_integer():
    x = torch.arange(8, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="floating-point"):
        frexp(x)


@pytest.mark.parametrize("shape", [(0,), (0, 7)])
def test_frexp_empty(shape):
    x = torch.randn(shape, dtype=torch.float32)
    mantissa, exponent = frexp(x)
    assert mantissa.shape == shape and exponent.shape == shape
    assert mantissa.numel() == 0 and exponent.numel() == 0
    assert exponent.dtype == torch.int32


def test_frexp_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(64, 32, dtype=torch.float32).t()
    assert not x.is_contiguous()
    _assert_frexp_close(x)


def test_frexp_triton_path(monkeypatch):
    # Smoke: the main path must launch the Triton kernel.
    import importlib

    mod = importlib.import_module("flaggems.frexp")

    launches = []
    real_run = mod.frexp_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.frexp_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32)
    mantissa, exponent = mod.frexp(x)
    assert len(launches) == 1, "frexp(A) must go through the Triton kernel"
    ref_m, ref_e = torch.frexp(x)
    torch.testing.assert_close(mantissa, ref_m, rtol=0.0, atol=0.0)
    torch.testing.assert_close(exponent, ref_e, rtol=0.0, atol=0.0)
