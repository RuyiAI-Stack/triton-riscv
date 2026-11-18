import pytest
import torch

from .and_tensor import and_tensor

# aten::__and__.Tensor is bitwise_and.Tensor: integer/bool operands only,
# output dtype is the promoted input dtype. For bool operands the bitwise
# AND coincides with the logical AND.
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
def test_and_tensor(shape, dtype):
    torch.manual_seed(0)
    x, y = _input_pair(shape, dtype)
    ref = torch.bitwise_and(x, y)
    tri = and_tensor(x, y)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri.dtype == ref.dtype


def test_and_tensor_bool_matches_logical_and():
    # For bool operands bitwise AND == logical AND (the B8 bool contract).
    torch.manual_seed(0)
    x, y = _input_pair((1024,), torch.bool)
    torch.testing.assert_close(
        and_tensor(x, y), torch.logical_and(x, y), rtol=0, atol=0
    )
    assert and_tensor(x, y).dtype == torch.bool


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_and_tensor_scalar(dtype):
    torch.manual_seed(0)
    x, _ = _input_pair((1024,), dtype)
    scalar = 0x55 if dtype is not torch.int8 else 0x5A
    ref = torch.bitwise_and(x, scalar)
    tri = and_tensor(x, scalar)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri.dtype == ref.dtype


def test_and_tensor_scalar_bool():
    torch.manual_seed(0)
    x = torch.rand(1024) < 0.5
    torch.testing.assert_close(
        and_tensor(x, True), torch.bitwise_and(x, True), rtol=0, atol=0
    )
    torch.testing.assert_close(
        and_tensor(x, False), torch.bitwise_and(x, False), rtol=0, atol=0
    )


def test_and_tensor_scalar_int64_extremes():
    # A full 64-bit mask scalar exercises the wide end of the scalar path.
    torch.manual_seed(0)
    x = torch.randint(-(2**63), 2**63 - 1, (1024,), dtype=torch.int64)
    mask = 0xF0F0F0F0F0F0F0F0
    torch.testing.assert_close(
        and_tensor(x, mask), torch.bitwise_and(x, mask), rtol=0, atol=0
    )


def test_and_tensor_extremes():
    torch.manual_seed(0)
    x = torch.tensor(
        [0, -1, 2**63 - 1, -(2**63), 0, -1],
        dtype=torch.int64,
    )
    y = torch.tensor(
        [-1, -1, 2**63 - 1, 2**63 - 1, 0, 0],
        dtype=torch.int64,
    )
    torch.testing.assert_close(
        and_tensor(x, y), torch.bitwise_and(x, y), rtol=0, atol=0
    )


def test_and_tensor_mixed_dtype_tensors():
    torch.manual_seed(0)
    x = torch.randint(-100, 100, (512,), dtype=torch.int32)
    y = torch.randint(-100, 100, (512,), dtype=torch.int64)
    tri = and_tensor(x, y)
    ref = torch.bitwise_and(x, y)
    assert tri.dtype == ref.dtype == torch.int64
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0,), (0, 37)])
def test_and_tensor_empty(shape):
    torch.manual_seed(0)
    x = torch.randint(-100, 100, shape, dtype=torch.int32)
    y = torch.randint(-100, 100, shape, dtype=torch.int32)
    torch.testing.assert_close(
        and_tensor(x, y), torch.bitwise_and(x, y), rtol=0, atol=0
    )
    torch.testing.assert_close(
        and_tensor(x, 1), torch.bitwise_and(x, 1), rtol=0, atol=0
    )


def test_and_tensor_non_contiguous():
    torch.manual_seed(0)
    x = torch.randint(-100, 100, (64, 32), dtype=torch.int32).t()
    y = torch.randint(-100, 100, (64, 32), dtype=torch.int32).t()
    assert not x.is_contiguous()
    torch.testing.assert_close(
        and_tensor(x, y), torch.bitwise_and(x, y), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((16, 256), (1, 256)),
        ((4, 128), (4, 1)),
        ((256,), (16, 256)),
    ],
)
def test_and_tensor_broadcast(shape_a, shape_b):
    torch.manual_seed(0)
    a = torch.randint(-100, 100, shape_a, dtype=torch.int32)
    b = torch.randint(-100, 100, shape_b, dtype=torch.int32)
    tri = and_tensor(a, b)
    ref = torch.bitwise_and(a, b)
    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_and_tensor_triton_path(monkeypatch):
    # Smoke: the tensor&tensor path must launch the Triton kernel.
    import importlib

    mod = importlib.import_module("flaggems.and_tensor")

    launches = []
    real_run = mod.and_tensor_kernel_tt.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.and_tensor_kernel_tt, "run", spy_run)
    x = torch.randint(-100, 100, (1024,), dtype=torch.int32)
    y = torch.randint(-100, 100, (1024,), dtype=torch.int32)
    out = mod.and_tensor(x, y)
    assert len(launches) == 1, "and_tensor(A, B) must go through the Triton kernel"
    torch.testing.assert_close(out, torch.bitwise_and(x, y), rtol=0, atol=0)


def test_and_tensor_scalar_triton_path(monkeypatch):
    # Smoke: the tensor&scalar overload must launch the scalar Triton kernel.
    import importlib

    mod = importlib.import_module("flaggems.and_tensor")

    launches = []
    real_run = mod.and_tensor_kernel_ts.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.and_tensor_kernel_ts, "run", spy_run)
    x = torch.randint(-100, 100, (1024,), dtype=torch.int32)
    out = mod.and_tensor(x, 0x55)
    assert len(launches) == 1, "and_tensor(A, scalar) must go through the Triton kernel"
    torch.testing.assert_close(out, torch.bitwise_and(x, 0x55), rtol=0, atol=0)
