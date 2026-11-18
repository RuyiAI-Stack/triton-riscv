import pytest
import torch

from .less_equal_ import less_equal_, less_equal_scalar_

B_DTYPES = [
    torch.bool,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.float32,
]


def _bool_self(shape):
    return torch.rand(shape) < 0.5


def _other(shape, dtype):
    if dtype == torch.bool:
        return torch.rand(shape) < 0.5
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype)
    return torch.randint(-2, 3, shape, dtype=dtype)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", B_DTYPES)
def test_less_equal__inplace(size, dtype):
    torch.manual_seed(0)
    a = _bool_self((size,))
    b = _other((size,), dtype)

    ref = a.clone()
    ref.less_equal_(b)

    tri = a.clone()
    out = less_equal_(tri, b)

    assert out is tri, "less_equal_ should return the same tensor"
    assert tri.dtype == torch.bool
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", B_DTYPES)
def test_less_equal_scalar_(dtype):
    torch.manual_seed(0)
    a = _bool_self((1024,))
    scalar = 1.5 if dtype.is_floating_point else 1

    ref = a.clone()
    ref.less_equal_(scalar)

    tri = a.clone()
    out = less_equal_scalar_(tri, scalar)

    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_less_equal_scalar__bool_scalar():
    torch.manual_seed(0)
    a = _bool_self((1024,))

    ref = a.clone()
    ref.less_equal_(True)

    tri = a.clone()
    out = less_equal_scalar_(tri, True)

    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_less_equal__inplace_non_contiguous():
    torch.manual_seed(0)
    base = _bool_self((64, 32))
    a = base.t()  # non-contiguous view
    b = torch.randn(32, 64)

    ref = a.clone()
    ref.less_equal_(b)

    tri = base.t()
    out = less_equal_(tri, b)

    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((16, 256), (1, 256)),
        ((4, 128), (4, 1)),
        ((16, 256), (16, 256)),
    ],
)
def test_less_equal__broadcast(shape_a, shape_b):
    torch.manual_seed(0)
    a = _bool_self(shape_a)
    b = _other(shape_b, torch.float32)

    ref = a.clone()
    ref.less_equal_(b)

    tri = a.clone()
    out = less_equal_(tri, b)

    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_less_equal__broadcast_wrong_shape_raises():
    # self smaller than the broadcast result is not writable in place.
    a = torch.zeros(16, 1, dtype=torch.bool)
    b = torch.randn(16, 256)
    with pytest.raises(RuntimeError):
        less_equal_(a, b)


def test_less_equal__non_bool_self_raises():
    # torch refuses to write a Bool result into a non-bool self.
    a = torch.zeros(1024, dtype=torch.int32)
    b = torch.zeros(1024, dtype=torch.int32)
    with pytest.raises(RuntimeError):
        less_equal_(a, b)


@pytest.mark.parametrize("shape", [(0,), (0, 37)])
def test_less_equal__empty(shape):
    torch.manual_seed(0)
    a = torch.empty(shape, dtype=torch.bool)
    b = torch.empty(shape, dtype=torch.float32)

    out = less_equal_(a, b)
    assert out is a
    assert a.shape == shape

    out_s = less_equal_scalar_(a.clone(), 1.0)
    assert out_s.shape == shape


def test_less_equal__triton_path(monkeypatch):
    # Smoke: the bool-self <= int32-tensor main path must launch the Triton
    # kernel.
    import importlib
    mod = importlib.import_module("flaggems.less_equal_")

    launches = []
    real_run = mod.less_equal_kernel__tt.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.less_equal_kernel__tt, "run", spy_run)
    a = torch.rand(1024) < 0.5
    b = torch.randint(-2, 3, (1024,), dtype=torch.int32)
    ref = a.clone()
    ref.less_equal_(b)

    out = mod.less_equal_(a, b)
    assert len(launches) == 1, "less_equal_(A, B) must go through the Triton kernel"
    assert out is a
    torch.testing.assert_close(a, ref, rtol=0, atol=0)


def test_less_equal_scalar__triton_path(monkeypatch):
    # Smoke: the in-place tensor<=scalar overload must launch the scalar
    # Triton kernel.
    import importlib
    mod = importlib.import_module("flaggems.less_equal_")

    launches = []
    real_run = mod.less_equal_kernel__ts.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.less_equal_kernel__ts, "run", spy_run)
    a = torch.rand(1024) < 0.5
    ref = a.clone()
    ref.less_equal_(1)

    out = mod.less_equal_scalar_(a, 1)
    assert (
        len(launches) == 1
    ), "less_equal_scalar_(A, B) must go through the Triton kernel"
    assert out is a
    torch.testing.assert_close(a, ref, rtol=0, atol=0)
