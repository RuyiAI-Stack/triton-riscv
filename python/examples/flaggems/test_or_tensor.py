import pytest
import torch

from .or_tensor import or_tensor

# aten::__or__.Tensor is bitwise_or.Tensor: integer/bool operands only,
# output dtype is the promoted input dtype. For bool operands the bitwise
# OR coincides with the logical OR.
BITWISE_DTYPES = [
    torch.bool,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]


def _input_pair(shape, dtype):
    if dtype is torch.bool:
        return (
            torch.rand(shape) < 0.5,
            torch.rand(shape) < 0.5,
        )
    return (
        torch.randint(-100, 100, shape, dtype=dtype),
        torch.randint(-100, 100, shape, dtype=dtype),
    )


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", BITWISE_DTYPES)
def test_or_tensor(shape, dtype):
    torch.manual_seed(0)
    x, y = _input_pair(shape, dtype)
    ref = torch.bitwise_or(x, y)
    tri = or_tensor(x, y)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri.dtype == ref.dtype


def test_or_tensor_bool_matches_logical_or():
    # For bool operands bitwise OR == logical OR (the B8 bool contract).
    torch.manual_seed(0)
    x, y = _input_pair((1024,), torch.bool)
    torch.testing.assert_close(
        or_tensor(x, y), torch.logical_or(x, y), rtol=0, atol=0
    )
    assert or_tensor(x, y).dtype == torch.bool


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_or_tensor_scalar(dtype):
    torch.manual_seed(0)
    x, _ = _input_pair((1024,), dtype)
    scalar = 0x55 if dtype is not torch.int8 else 0x5A
    ref = torch.bitwise_or(x, scalar)
    tri = or_tensor(x, scalar)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri.dtype == ref.dtype


def test_or_tensor_scalar_bool():
    torch.manual_seed(0)
    x = torch.rand(1024) < 0.5
    torch.testing.assert_close(
        or_tensor(x, True), torch.bitwise_or(x, True), rtol=0, atol=0
    )
    torch.testing.assert_close(
        or_tensor(x, False), torch.bitwise_or(x, False), rtol=0, atol=0
    )


def test_or_tensor_scalar_int64_extremes():
    # A full 64-bit mask scalar exercises the wide end of the scalar path.
    torch.manual_seed(0)
    x = torch.randint(-2**63, 2**63 - 1, (1024,), dtype=torch.int64)
    mask = 0xF0F0F0F0F0F0F0F0
    torch.testing.assert_close(
        or_tensor(x, mask), torch.bitwise_or(x, mask), rtol=0, atol=0
    )


def test_or_tensor_extremes():
    torch.manual_seed(0)
    x = torch.tensor(
        [0, -1, 2**63 - 1, -2**63, 0, -1],
        dtype=torch.int64,
    )
    y = torch.tensor(
        [-1, -1, 2**63 - 1, 2**63 - 1, 0, 0],
        dtype=torch.int64,
    )
    torch.testing.assert_close(
        or_tensor(x, y), torch.bitwise_or(x, y), rtol=0, atol=0
    )


def test_or_tensor_mixed_dtype_tensors():
    torch.manual_seed(0)
    x = torch.randint(-100, 100, (512,), dtype=torch.int32)
    y = torch.randint(-100, 100, (512,), dtype=torch.int64)
    tri = or_tensor(x, y)
    ref = torch.bitwise_or(x, y)
    assert tri.dtype == ref.dtype == torch.int64
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0,), (0, 37)])
def test_or_tensor_empty(shape):
    torch.manual_seed(0)
    x = torch.randint(-100, 100, shape, dtype=torch.int32)
    y = torch.randint(-100, 100, shape, dtype=torch.int32)
    torch.testing.assert_close(
        or_tensor(x, y), torch.bitwise_or(x, y), rtol=0, atol=0
    )
    torch.testing.assert_close(
        or_tensor(x, 1), torch.bitwise_or(x, 1), rtol=0, atol=0
    )


def test_or_tensor_non_contiguous():
    torch.manual_seed(0)
    x = torch.randint(-100, 100, (64, 32), dtype=torch.int32).t()
    y = torch.randint(-100, 100, (64, 32), dtype=torch.int32).t()
    assert not x.is_contiguous()
    torch.testing.assert_close(
        or_tensor(x, y), torch.bitwise_or(x, y), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((16, 256), (1, 256)),
        ((4, 128), (4, 1)),
        ((256,), (16, 256)),
    ],
)
def test_or_tensor_broadcast(shape_a, shape_b):
    torch.manual_seed(0)
    a = torch.randint(-100, 100, shape_a, dtype=torch.int32)
    b = torch.randint(-100, 100, shape_b, dtype=torch.int32)
    tri = or_tensor(a, b)
    ref = torch.bitwise_or(a, b)
    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_or_tensor_triton_path(monkeypatch):
    # Smoke: the tensor|tensor path must launch the Triton kernel.
    import importlib
    mod = importlib.import_module("flaggems.or_tensor")

    launches = []
    real_run = mod.or_tensor_kernel_tt.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.or_tensor_kernel_tt, "run", spy_run)
    x = torch.randint(-100, 100, (1024,), dtype=torch.int32)
    y = torch.randint(-100, 100, (1024,), dtype=torch.int32)
    out = mod.or_tensor(x, y)
    assert len(launches) == 1, "or_tensor(A, B) must go through the Triton kernel"
    torch.testing.assert_close(out, torch.bitwise_or(x, y), rtol=0, atol=0)


def test_or_tensor_scalar_triton_path(monkeypatch):
    # Smoke: the tensor|scalar overload must launch the scalar Triton kernel.
    import importlib
    mod = importlib.import_module("flaggems.or_tensor")

    launches = []
    real_run = mod.or_tensor_kernel_ts.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.or_tensor_kernel_ts, "run", spy_run)
    x = torch.randint(-100, 100, (1024,), dtype=torch.int32)
    out = mod.or_tensor(x, 0x55)
    assert (
        len(launches) == 1
    ), "or_tensor(A, scalar) must go through the Triton kernel"
    torch.testing.assert_close(out, torch.bitwise_or(x, 0x55), rtol=0, atol=0)
