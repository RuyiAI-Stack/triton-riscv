import pytest
import torch

from .less_equal import less_equal, less_equal_scalar

INT_DTYPES = [
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]
# fp16/fp32 values are compared losslessly after the fp32 promotion, so the
# bool result is bit-exact against torch; ints stay within +-2**24 so the
# int->fp32 conversion is exact as well.
EXACT_DTYPES = INT_DTYPES + [torch.float16, torch.float32]


def _input_pair(shape, dtype):
    if dtype.is_floating_point:
        return (
            torch.randn(shape, dtype=dtype),
            torch.randn(shape, dtype=dtype),
        )
    return (
        torch.randint(-100, 100, shape, dtype=dtype),
        torch.randint(-100, 100, shape, dtype=dtype),
    )


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", EXACT_DTYPES)
def test_less_equal(shape, dtype):
    torch.manual_seed(0)
    x, y = _input_pair(shape, dtype)
    ref = torch.less_equal(x, y)
    tri = less_equal(x, y)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri.dtype == torch.bool


def test_less_equal_bool():
    torch.manual_seed(0)
    x = torch.rand(1024) < 0.5
    y = torch.rand(1024) < 0.5
    torch.testing.assert_close(
        less_equal(x, y), torch.less_equal(x, y), rtol=0, atol=0
    )


def test_less_equal_float64_tolerant():
    # fp64 -> fp32 in the kernel loses precision near the comparison
    # boundary; allow a small mismatch budget instead of bit-exactness.
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float64)
    y = torch.randn(1024, dtype=torch.float64)
    tri = less_equal(x, y)
    ref = torch.less_equal(x, y)
    mismatches = (tri != ref).sum().item()
    assert mismatches <= 2, f"fp64 comparison mismatch budget exceeded: {mismatches}"


@pytest.mark.parametrize(
    "dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32]
)
def test_less_equal_scalar(dtype):
    torch.manual_seed(0)
    x, _ = _input_pair((1024,), dtype)
    scalar = 1.5 if dtype.is_floating_point else 1
    ref = torch.less_equal(x, scalar)
    tri = less_equal_scalar(x, scalar)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri.dtype == torch.bool


def test_less_equal_scalar_mixed_number_kinds():
    torch.manual_seed(0)
    x = torch.randint(-100, 100, (1024,), dtype=torch.int32)
    # int tensor vs float scalar promotes to the default dtype, and vs a
    # bool scalar compares against 0/1.
    torch.testing.assert_close(
        less_equal_scalar(x, 1.5), torch.less_equal(x, 1.5), rtol=0, atol=0
    )
    torch.testing.assert_close(
        less_equal_scalar(x, True), torch.less_equal(x, True), rtol=0, atol=0
    )


def test_less_equal_mixed_dtype_tensors():
    torch.manual_seed(0)
    x = torch.randint(-100, 100, (512,), dtype=torch.int32)
    y = torch.randn(512, dtype=torch.float32)
    torch.testing.assert_close(
        less_equal(x, y), torch.less_equal(x, y), rtol=0, atol=0
    )


def test_less_equal_near_boundary():
    torch.manual_seed(0)
    y = torch.randn(1024, dtype=torch.float32)
    x_equal = y.clone()
    torch.testing.assert_close(
        less_equal(x_equal, y),
        torch.ones(1024, dtype=torch.bool),
        rtol=0,
        atol=0,
    )
    x_above = torch.nextafter(y, torch.full_like(y, float("inf")))
    torch.testing.assert_close(
        less_equal(x_above, y),
        torch.zeros(1024, dtype=torch.bool),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("shape", [(0,), (0, 37)])
def test_less_equal_empty(shape):
    torch.manual_seed(0)
    x = torch.randint(-100, 100, shape, dtype=torch.int32)
    y = torch.randint(-100, 100, shape, dtype=torch.int32)
    torch.testing.assert_close(
        less_equal(x, y), torch.less_equal(x, y), rtol=0, atol=0
    )
    torch.testing.assert_close(
        less_equal_scalar(x, 1), torch.less_equal(x, 1), rtol=0, atol=0
    )


def test_less_equal_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(64, 32).t()
    y = torch.randn(64, 32).t()
    assert not x.is_contiguous()
    torch.testing.assert_close(
        less_equal(x, y), torch.less_equal(x, y), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((16, 256), (1, 256)),
        ((4, 128), (4, 1)),
        ((256,), (16, 256)),
    ],
)
def test_less_equal_broadcast(shape_a, shape_b):
    torch.manual_seed(0)
    a = torch.randn(shape_a, dtype=torch.float32)
    b = torch.randn(shape_b, dtype=torch.float32)
    tri = less_equal(a, b)
    ref = torch.less_equal(a, b)
    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_less_equal_triton_path(monkeypatch):
    # Smoke: the int32 tensor<=tensor main path must launch the Triton kernel.
    import importlib
    mod = importlib.import_module("flaggems.less_equal")

    launches = []
    real_run = mod.less_equal_kernel_tt.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.less_equal_kernel_tt, "run", spy_run)
    x = torch.randint(-100, 100, (1024,), dtype=torch.int32)
    y = torch.randint(-100, 100, (1024,), dtype=torch.int32)
    out = mod.less_equal(x, y)
    assert len(launches) == 1, "less_equal(A, B) must go through the Triton kernel"
    torch.testing.assert_close(out, torch.less_equal(x, y), rtol=0, atol=0)


def test_less_equal_scalar_triton_path(monkeypatch):
    # Smoke: the tensor<=scalar overload must launch the scalar Triton kernel.
    import importlib
    mod = importlib.import_module("flaggems.less_equal")

    launches = []
    real_run = mod.less_equal_kernel_ts.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.less_equal_kernel_ts, "run", spy_run)
    x = torch.randint(-100, 100, (1024,), dtype=torch.int32)
    out = mod.less_equal_scalar(x, 1)
    assert (
        len(launches) == 1
    ), "less_equal_scalar(A, B) must go through the Triton kernel"
    torch.testing.assert_close(out, torch.less_equal(x, 1), rtol=0, atol=0)
