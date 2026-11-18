import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def tensor_gather_kernel(
    X,
    Indices,
    Y,
    AXIS: tl.constexpr,
    IR: tl.constexpr,
    IC: tl.constexpr,
    REPEATS: tl.constexpr,
):
    source = tl.load(X + tl.arange(0, 4)[:, None] * 8 + tl.arange(0, 8)[None, :])
    offsets = tl.arange(0, IR)[:, None] * IC + tl.arange(0, IC)[None, :]
    indices = tl.load(Indices + offsets)
    for _ in tl.static_range(REPEATS):
        source = tl.gather(source, indices, AXIS)
    tl.store(Y + offsets, source)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "axis,shape,repeats",
    [(0, (2, 8), 1), (1, (4, 4), 1), (0, (4, 8), 2), (1, (4, 8), 2)],
)
def test_tensor_gather(dtype, index_dtype, axis, shape, repeats):
    torch.manual_seed(42)
    x = torch.arange(32).reshape(4, 8).to(dtype)
    indices = torch.randint(x.shape[axis], shape, dtype=index_dtype)
    expected = x
    for _ in range(repeats):
        expected = torch.gather(expected, axis, indices.long())
    actual = torch.empty(shape, dtype=dtype)
    tensor_gather_kernel[(1,)](x, indices, actual, axis, *shape, repeats)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@triton.jit
def tensor_gather_loop_kernel(X, Indices, Y, repeats, AXIS: tl.constexpr):
    offsets = tl.arange(0, 4)[:, None] * 8 + tl.arange(0, 8)[None, :]
    source = tl.load(X + offsets)
    indices = tl.load(Indices + offsets)
    for _ in range(repeats):
        source = tl.gather(source, indices, AXIS)
    tl.store(Y + offsets, source)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("repeats", [0, 3])
def test_tensor_gather_loop_carried(axis, repeats):
    torch.manual_seed(7)
    x = torch.arange(32, dtype=torch.float64).reshape(4, 8)
    indices = torch.randint(x.shape[axis], x.shape, dtype=torch.int32)
    expected = x
    for _ in range(repeats):
        expected = torch.gather(expected, axis, indices.long())
    actual = torch.empty_like(x)
    tensor_gather_loop_kernel[(1,)](x, indices, actual, repeats, axis)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@triton.jit
def tensor_gather_narrow_indices_kernel(X, Indices, Y, N: tl.constexpr):
    source = tl.load(X + tl.arange(0, N))
    offsets = tl.arange(0, 4)
    indices = tl.load(Indices + offsets)
    result = tl.gather(source, indices, 0)
    tl.store(Y + offsets, result)


@pytest.mark.parametrize(
    "index_dtype,n,values",
    [
        (torch.bool, 2, [False, True, True, False]),
        (torch.uint8, 256, [0, 127, 128, 255]),
    ],
)
def test_tensor_gather_narrow_indices(index_dtype, n, values):
    x = torch.arange(n, dtype=torch.float32)
    indices = torch.tensor(values, dtype=index_dtype)
    expected = torch.gather(x, 0, indices.long())
    actual = torch.empty(indices.shape, dtype=x.dtype)
    tensor_gather_narrow_indices_kernel[(1,)](x, indices, actual, n)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
