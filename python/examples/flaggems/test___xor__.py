import pytest
import torch

from .__xor__ import xor, xor_, xor_scalar, xor_scalar_, xor_scalar_tensor

INT_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
]


def _randint(shape, dtype):
    if dtype == torch.int8:
        return torch.randint(-128, 128, shape, dtype=dtype)
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=dtype)
    if dtype == torch.int16:
        return torch.randint(-32768, 32768, shape, dtype=dtype)
    return torch.randint(-100000, 100000, shape, dtype=dtype)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_xor_tensor(shape, dtype):
    torch.manual_seed(0)
    x = _randint(shape, dtype)
    y = _randint(shape, dtype)
    ref = torch.bitwise_xor(x, y)
    tri = xor(x, y)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri.dtype == ref.dtype


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_xor_tensor_bool(shape):
    torch.manual_seed(0)
    x = torch.rand(shape) < 0.5
    y = torch.rand(shape) < 0.5
    ref = torch.bitwise_xor(x, y)
    tri = xor(x, y)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri.dtype == torch.bool


@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_xor_tensor_(dtype):
    torch.manual_seed(0)
    x = _randint((4, 128), dtype)
    y = _randint((4, 128), dtype)
    ref = torch.bitwise_xor(x.clone(), y)
    x_copy = x.clone()
    out = xor_(x_copy, y)
    assert out is x_copy
    torch.testing.assert_close(x_copy, ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_xor_scalar(dtype):
    torch.manual_seed(0)
    x = _randint((1024,), dtype)
    ref = torch.bitwise_xor(x, 0x5)
    tri = xor_scalar(x, 0x5)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri.dtype == x.dtype


def test_xor_scalar_bool_tensor_and_scalar():
    torch.manual_seed(0)
    x = torch.rand(1024) < 0.5
    ref = torch.bitwise_xor(x, True)
    tri = xor_scalar(x, True)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri.dtype == torch.bool


def test_xor_scalar_():
    torch.manual_seed(0)
    x = torch.randint(0, 255, (1024,), dtype=torch.int32)
    ref = torch.bitwise_xor(x.clone(), 0x5)
    x_copy = x.clone()
    out = xor_scalar_(x_copy, 0x5)
    assert out is x_copy
    torch.testing.assert_close(x_copy, ref, rtol=0, atol=0)


def test_xor_scalar_tensor():
    torch.manual_seed(0)
    y = torch.randint(0, 255, (1024,), dtype=torch.int32)
    ref = torch.bitwise_xor(0xF0, y)
    tri = xor_scalar_tensor(0xF0, y)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri.dtype == torch.int32


def test_xor_scalar_tensor_bool():
    torch.manual_seed(0)
    y = torch.rand(1024) < 0.5
    ref = torch.bitwise_xor(True, y)
    tri = xor_scalar_tensor(True, y)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0,), (0, 37)])
def test_xor_empty(shape):
    torch.manual_seed(0)
    x = torch.randint(0, 255, shape, dtype=torch.int32)
    y = torch.randint(0, 255, shape, dtype=torch.int32)
    torch.testing.assert_close(xor(x, y), torch.bitwise_xor(x, y), rtol=0, atol=0)
    torch.testing.assert_close(
        xor_scalar(x, 0x5), torch.bitwise_xor(x, 0x5), rtol=0, atol=0
    )
    torch.testing.assert_close(
        xor_scalar_tensor(0x5, y), torch.bitwise_xor(0x5, y), rtol=0, atol=0
    )


def test_xor_non_contiguous():
    torch.manual_seed(0)
    x = torch.randint(0, 255, (64, 32), dtype=torch.int32).t()
    y = torch.randint(0, 255, (64, 32), dtype=torch.int32).t()
    assert not x.is_contiguous()
    ref = torch.bitwise_xor(x, y)
    tri = xor(x, y)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_xor_non_contiguous_inplace():
    torch.manual_seed(0)
    base = torch.randint(0, 255, (64, 32), dtype=torch.int32)
    y = torch.randint(0, 255, (64, 32), dtype=torch.int32).t()
    ref = torch.bitwise_xor(base.t().clone(), y)
    tri = base.t()  # non-contiguous view
    out = xor_(tri, y)
    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((16, 256), (1, 256)),
        ((4, 128), (4, 1)),
        ((256,), (16, 256)),
    ],
)
def test_xor_broadcast(shape_a, shape_b):
    torch.manual_seed(0)
    a = torch.randint(0, 255, shape_a, dtype=torch.int32)
    b = torch.randint(0, 255, shape_b, dtype=torch.int32)
    ref = torch.bitwise_xor(a, b)
    tri = xor(a, b)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri.shape == ref.shape


def test_xor_negative_values():
    torch.manual_seed(0)
    x = torch.randint(-128, 128, (1024,), dtype=torch.int32)
    y = torch.randint(-128, 128, (1024,), dtype=torch.int32)
    torch.testing.assert_close(xor(x, y), torch.bitwise_xor(x, y), rtol=0, atol=0)


def test_xor_mixed_int_promotes():
    # torch promotes (int32, int64) -> int64 without truncating.
    x = torch.tensor([5, -1], dtype=torch.int32)  # -1 == 0xFFFFFFFF bits
    y = torch.tensor([3, 1], dtype=torch.int64)
    out = xor(x, y)
    assert out.dtype == torch.int64
    torch.testing.assert_close(out, torch.bitwise_xor(x, y), rtol=0, atol=0)


def test_xor_inplace_small_out_raises():
    # Broadcastable shapes with a smaller in-place target must raise instead
    # of launching with the broadcast numel over a smaller buffer.
    a = torch.zeros(16, 1, dtype=torch.int32)
    b = torch.ones(16, 256, dtype=torch.int32)
    with pytest.raises(ValueError):
        xor_(a, b)


def test_xor_triton_path(monkeypatch):
    # Smoke: the int32 tensor^tensor main path must launch the Triton kernel.
    from . import __xor__ as mod

    launches = []
    real_run = mod.xor_kernel_tt.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.xor_kernel_tt, "run", spy_run)
    x = torch.randint(0, 255, (1024,), dtype=torch.int32)
    y = torch.randint(0, 255, (1024,), dtype=torch.int32)
    out = mod.xor(x, y)
    assert len(launches) == 1, "xor(A, B) must go through the Triton kernel"
    torch.testing.assert_close(out, torch.bitwise_xor(x, y), rtol=0, atol=0)


def test_xor_scalar_triton_path(monkeypatch):
    # Smoke: the tensor^scalar overload must launch the scalar Triton kernel.
    from . import __xor__ as mod

    launches = []
    real_run = mod.xor_kernel_ts.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.xor_kernel_ts, "run", spy_run)
    x = torch.randint(0, 255, (1024,), dtype=torch.int32)
    out = mod.xor_scalar(x, 0x5)
    assert len(launches) == 1, "xor_scalar(A, B) must go through the Triton kernel"
    torch.testing.assert_close(out, torch.bitwise_xor(x, 0x5), rtol=0, atol=0)
