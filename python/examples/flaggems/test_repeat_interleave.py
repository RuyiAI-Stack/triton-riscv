import pytest
import torch

from .repeat_interleave import (
    repeat_interleave,
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)


@pytest.mark.parametrize(
    "shape, repeats, dim",
    [
        ((5,), 2, None),
        ((2, 3), 2, 0),
        ((2, 3), 3, 1),
        ((2, 3), 3, -1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_repeat_interleave_int(shape, repeats, dim, dtype):
    x = torch.randn(shape).to(dtype)

    out_triton = repeat_interleave(x, repeats, dim)
    out_torch = torch.repeat_interleave(x, repeats, dim)

    torch.testing.assert_close(out_triton, out_torch)


@pytest.mark.parametrize(
    "shape, repeats_list, dim",
    [
        ((3,), [1, 2, 3], None),
        ((2, 3), [2, 1], 0),
        ((2, 3), [1, 3, 2], 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_repeat_interleave_tensor(shape, repeats_list, dim, dtype):
    x = torch.randn(shape).to(dtype)
    repeats = torch.tensor(repeats_list, dtype=torch.int64)

    out_triton = repeat_interleave(x, repeats, dim)
    out_torch = torch.repeat_interleave(x, repeats, dim)

    torch.testing.assert_close(out_triton, out_torch)


@pytest.mark.parametrize(
    "shape, repeats, dim",
    [
        ((5,), 2, None),
        ((2, 3), 2, 0),
        ((2, 3), 3, 1),
        ((2, 3), 3, -1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_repeat_interleave_self_int_func(shape, repeats, dim, dtype):
    x = torch.randn(shape).to(dtype)

    out_triton = repeat_interleave_self_int(x, repeats, dim)
    out_torch = torch.repeat_interleave(x, repeats, dim)

    torch.testing.assert_close(out_triton, out_torch)


@pytest.mark.parametrize(
    "shape, repeats_list, dim",
    [
        ((3,), [1, 2, 3], None),
        ((2, 3), [2, 1], 0),
        ((2, 3), [1, 3, 2], 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_repeat_interleave_self_tensor_func(shape, repeats_list, dim, dtype):
    x = torch.randn(shape).to(dtype)
    repeats = torch.tensor(repeats_list, dtype=torch.int64)

    out_triton = repeat_interleave_self_tensor(x, repeats, dim)
    out_torch = torch.repeat_interleave(x, repeats, dim)

    torch.testing.assert_close(out_triton, out_torch)


@pytest.mark.parametrize("repeats_list", [[1, 2, 3], [3, 0, 1, 2], [5, 0, 0, 1]])
def test_repeat_interleave_tensor_func(repeats_list):
    repeats = torch.tensor(repeats_list, dtype=torch.int64)

    out_triton = repeat_interleave_tensor(repeats)
    out_torch = torch.repeat_interleave(repeats)

    torch.testing.assert_close(out_triton, out_torch)
