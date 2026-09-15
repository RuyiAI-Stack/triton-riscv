import pytest
import torch

from .searchsorted import (
    searchsorted,
    searchsorted_out,
    searchsorted_scalar,
    searchsorted_scalar_out,
)


@pytest.mark.parametrize("shape", [(16,), (128,), (1023,)])
@pytest.mark.parametrize("val_shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.float64])
def test_searchsorted(shape, val_shape, dtype):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        boundaries = torch.sort(torch.randn(shape, dtype=dtype, device="cpu"))[0]
        values = torch.randn(val_shape, dtype=dtype, device="cpu")
    else:
        boundaries = torch.sort(
            torch.randint(-100, 100, shape, dtype=dtype, device="cpu")
        )[0]
        values = torch.randint(-120, 120, val_shape, dtype=dtype, device="cpu")

    tri_out = searchsorted(boundaries, values, right=False)
    ref_out = torch.searchsorted(boundaries, values, right=False)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(16,), (128,), (1023,)])
@pytest.mark.parametrize("val_shape", [(16,), (4,), (512,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_searchsorted_out_and_scalar(shape, val_shape, dtype):
    torch.manual_seed(0)
    boundaries = torch.sort(torch.randn(shape, dtype=dtype, device="cpu"))[0]
    values = torch.randn(val_shape, dtype=dtype, device="cpu")
    tri_out = torch.empty(val_shape, dtype=torch.int64, device="cpu")

    ret = searchsorted_out(boundaries, values, out=tri_out)

    # Check scalar
    scalar_val = torch.randn((), dtype=dtype, device="cpu").item()
    scalar = searchsorted_scalar(boundaries, scalar_val)

    assert ret is tri_out
    torch.testing.assert_close(tri_out, torch.searchsorted(boundaries, values))
    torch.testing.assert_close(
        scalar,
        torch.searchsorted(
            boundaries, torch.tensor(scalar_val, dtype=dtype, device="cpu")
        ),
    )


@pytest.mark.parametrize("shape", [(16,), (128,), (1023,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_searchsorted_scalar_out(shape, dtype):
    torch.manual_seed(0)
    boundaries = torch.sort(torch.randn(shape, dtype=dtype, device="cpu"))[0]
    tri_out = torch.empty((), dtype=torch.int64, device="cpu")
    scalar_val = torch.randn((), dtype=dtype, device="cpu").item()
    ref_out = torch.empty((), dtype=torch.int64, device="cpu")

    ret = searchsorted_scalar_out(boundaries, scalar_val, out=tri_out)
    torch.searchsorted(
        boundaries,
        torch.tensor(scalar_val, dtype=dtype, device="cpu"),
        out=ref_out,
    )

    torch.testing.assert_close(tri_out, ref_out)
    assert ret is tri_out


def test_searchsorted_right_and_side():
    boundaries = torch.tensor([1.0, 3.0, 3.0, 5.0, 8.0], device="cpu")
    values = torch.tensor([0.0, 3.0, 4.0, 9.0], device="cpu")

    torch.testing.assert_close(
        searchsorted(boundaries, values, right=False),
        torch.searchsorted(boundaries, values, right=False),
    )
    torch.testing.assert_close(
        searchsorted(boundaries, values, right=True),
        torch.searchsorted(boundaries, values, right=True),
    )
    torch.testing.assert_close(
        searchsorted(boundaries, values, side="left"),
        torch.searchsorted(boundaries, values, side="left"),
    )
    torch.testing.assert_close(
        searchsorted(boundaries, values, side="right"),
        torch.searchsorted(boundaries, values, side="right"),
    )
    torch.testing.assert_close(
        searchsorted_scalar(boundaries, 3.0, right=True),
        torch.searchsorted(boundaries, torch.tensor(3.0, device="cpu"), right=True),
    )


def test_searchsorted_sorter():
    boundaries = torch.tensor([30.0, 10.0, 20.0], device="cpu")
    sorter = torch.tensor([1, 2, 0], dtype=torch.int64, device="cpu")
    values = torch.tensor([5.0, 15.0, 25.0, 35.0], device="cpu")

    torch.testing.assert_close(
        searchsorted(boundaries, values, sorter=sorter),
        torch.searchsorted(boundaries, values, sorter=sorter),
    )


def test_searchsorted_nd():
    boundaries = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], device="cpu")
    values = torch.tensor([[0.0, 4.0, 7.0], [1.0, 3.0, 6.0]], device="cpu")

    torch.testing.assert_close(
        searchsorted(boundaries, values),
        torch.searchsorted(boundaries, values),
    )


def test_searchsorted_out_int32_non_contiguous():
    boundaries = torch.tensor([1.0, 3.0, 3.0, 5.0, 8.0], device="cpu")
    values = torch.tensor([[0.0, 3.0, 4.0], [9.0, 1.0, 5.0]], device="cpu")
    out = torch.empty((2, 6), dtype=torch.int32, device="cpu")[:, ::2]

    ret = searchsorted_out(boundaries, values, out_int32=True, right=True, out=out)
    ref = torch.searchsorted(boundaries, values, out_int32=True, right=True)

    assert ret is out
    torch.testing.assert_close(out, ref)


def test_searchsorted_empty_inputs():
    boundaries = torch.empty((0,), dtype=torch.float32, device="cpu")
    values = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32, device="cpu")

    torch.testing.assert_close(
        searchsorted(boundaries, values), torch.searchsorted(boundaries, values)
    )
    empty_values = torch.empty((0,), dtype=torch.float32, device="cpu")
    torch.testing.assert_close(
        searchsorted(torch.tensor([1.0], device="cpu"), empty_values),
        torch.searchsorted(torch.tensor([1.0], device="cpu"), empty_values),
    )


def test_searchsorted_nd_sorter():
    boundaries = torch.tensor([[30.0, 10.0, 20.0], [6.0, 2.0, 4.0]], device="cpu")
    sorter = torch.tensor([[1, 2, 0], [1, 2, 0]], dtype=torch.int64, device="cpu")
    values = torch.tensor([[5.0, 15.0, 25.0], [1.0, 3.0, 7.0]], device="cpu")

    torch.testing.assert_close(
        searchsorted(boundaries, values, sorter=sorter),
        torch.searchsorted(boundaries, values, sorter=sorter),
    )


def test_searchsorted_rejects_conflicting_side_and_right():
    boundaries = torch.tensor([1.0, 2.0], device="cpu")
    values = torch.tensor([1.5], device="cpu")

    with pytest.raises(RuntimeError):
        torch.searchsorted(boundaries, values, right=True, side="left")
    with pytest.raises(RuntimeError):
        searchsorted(boundaries, values, right=True, side="left")
