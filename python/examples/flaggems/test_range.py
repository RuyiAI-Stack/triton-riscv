import pytest
import torch

# NOTE: `range` here is the op under test (deprecated torch.range semantics),
# so the builtin range() must not be used anywhere in this module.
from .range import range


@pytest.mark.parametrize("end", [511, 1022, 1023])
def test_range_int_bounds(end):
    x = range(0, end)
    ref = torch.range(0, end)
    assert x.dtype == torch.get_default_dtype() == ref.dtype
    assert x.shape == ref.shape
    # Integer-valued fp32 elements are exact on both sides.
    torch.testing.assert_close(x, ref, rtol=0.0, atol=0.0)


def test_range_default_dtype_is_float32():
    # torch.range (deprecated) always defaults to the default float dtype,
    # even for integer bounds -- unlike torch.arange, which picks int64.
    assert range(0, 4).dtype == torch.float32
    assert range(1.5, 3.5).dtype == torch.float32


def test_range_contains_endpoint():
    torch.manual_seed(0)
    for step in (1, 2, 7):
        x = range(3, 100, step)
        ref = torch.range(3, 100, step)
        assert x.numel() == ref.numel()
        torch.testing.assert_close(x, ref, rtol=0.0, atol=0.0)

    x = range(1.5, 3.5, 0.5)
    ref = torch.range(1.5, 3.5, 0.5)
    assert x[-1].item() == 3.5, "the end point is included"
    torch.testing.assert_close(x, ref, rtol=0.0, atol=0.0)


def test_range_negative_step():
    x = range(5, 1, -1)
    ref = torch.range(5, 1, -1)
    assert x.tolist() == [5.0, 4.0, 3.0, 2.0, 1.0]
    torch.testing.assert_close(x, ref, rtol=0.0, atol=0.0)

    x2 = range(-3, -7, -2)
    ref2 = torch.range(-3, -7, -2)
    assert x2.tolist() == [-3.0, -5.0, -7.0]
    torch.testing.assert_close(x2, ref2, rtol=0.0, atol=0.0)


def test_range_negative_bounds_positive_step():
    x = range(-7, -2)
    ref = torch.range(-7, -2)
    torch.testing.assert_close(x, ref, rtol=0.0, atol=0.0)


def test_range_float_steps_representable():
    x = range(-2.5, -0.5, 0.5)
    ref = torch.range(-2.5, -0.5, 0.5)
    torch.testing.assert_close(x, ref, rtol=0.0, atol=0.0)


def test_range_float_steps_non_representable():
    # 0.1/0.3 are not exactly representable: torch computes each element in
    # scalar_t and the element count in double; the counts must match
    # exactly and the values within fp32 rounding of torch's.
    for start, end, step in [(0.1, 0.7, 0.1), (0.0, 1.0, 0.1), (1.0, 2.0, 0.3)]:
        x = range(start, end, step)
        ref = torch.range(start, end, step)
        assert x.numel() == ref.numel()
        torch.testing.assert_close(x, ref, rtol=1e-5, atol=1e-6)

    # Non-integral span: 11 elements [0 .. 10], endpoint 10.5 excluded.
    x = range(0, 10.5)
    ref = torch.range(0, 10.5)
    assert x.numel() == 11 and ref.numel() == 11
    assert x[-1].item() == 10.0
    torch.testing.assert_close(x, ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_range_explicit_integer_dtypes(dtype):
    x = range(-7, 1000, 3, dtype=dtype)
    ref = torch.range(-7, 1000, 3, dtype=dtype)
    assert x.dtype == dtype
    torch.testing.assert_close(x, ref, rtol=0.0, atol=0.0)


def test_range_explicit_float64():
    # Exactly representable start/step keep the fp32 kernel arithmetic exact.
    x = range(1.5, 257.5, 0.5, dtype=torch.float64)
    ref = torch.range(1.5, 257.5, 0.5, dtype=torch.float64)
    assert x.dtype == torch.float64
    torch.testing.assert_close(x, ref, rtol=0.0, atol=0.0)


def test_range_single_element():
    assert range(3, 3).tolist() == [3.0]
    assert range(2.5, 2.5).tolist() == [2.5]


def test_range_bf16_rejected():
    # Upstream mirror of torch.range's CUDA-side bfloat16 rejection.
    with pytest.raises(RuntimeError, match="bfloat16"):
        range(0, 4, dtype=torch.bfloat16)


def test_range_step_zero():
    with pytest.raises(RuntimeError, match="step must be nonzero"):
        range(0, 4, 0)


@pytest.mark.parametrize(
    "start,end,step",
    [(5, 1, 1), (1, 5, -1), (5.0, 1.0, 2.0), (1.0, 5.0, -0.5)],
)
def test_range_sign_mismatch(start, end, step):
    with pytest.raises(RuntimeError, match="upper bound and lower bound"):
        range(start, end, step)


def test_range_triton_path(monkeypatch):
    # Smoke: the main path must launch the Triton kernel (1024 elements =
    # 8 blocks of BLOCK_SIZE=128, exercising the tail mask).
    import importlib

    mod = importlib.import_module("flaggems.range")

    launches = []
    real_run = mod.range_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.range_func, "run", spy_run)
    out = mod.range(0, 1023)
    assert len(launches) == 1 and launches[0] == (8,)
    torch.testing.assert_close(out, torch.range(0, 1023), rtol=0.0, atol=0.0)
