import pytest
import torch

from .div import (
    div,
    div_mode_,
    floor_divide,
    floor_divide_,
    remainder,
    remainder_,
    true_divide,
    true_divide_,
    true_divide_out,
)


# =============================================================================
# Original tests: div (alias for div_mode) and remainder
# =============================================================================


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512, 64), (1023, 64), (1024, 64)],
)
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_div_mode_tt(shape, rounding_mode, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = torch.rand(shape, dtype=dtype, device="cpu") * 5 + 1

    ref_out = torch.div(x, y, rounding_mode=rounding_mode)
    tri_out = div(x, y, rounding_mode=rounding_mode)

    if dtype == torch.float16 and rounding_mode == "trunc":
        rtol, atol = 1e-2, 1.0
    else:
        rtol, atol = 1e-3, 1e-3
    torch.testing.assert_close(tri_out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_div_mode_ts(shape, rounding_mode, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = 5.0

    ref_out = torch.div(x, y, rounding_mode=rounding_mode)
    tri_out = div(x, y, rounding_mode=rounding_mode)

    if dtype == torch.float16 and rounding_mode == "trunc":
        rtol, atol = 1e-2, 1.0
    else:
        rtol, atol = 1e-3, 1e-3
    torch.testing.assert_close(tri_out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_div_mode_st(shape, rounding_mode, dtype):
    torch.manual_seed(0)
    x = 5.0
    y = torch.rand(shape, dtype=dtype, device="cpu") * 5 + 1

    ref_out = torch.div(x, y, rounding_mode=rounding_mode)
    tri_out = div(x, y, rounding_mode=rounding_mode)

    if dtype == torch.float16 and rounding_mode == "trunc":
        rtol, atol = 1e-2, 1.0
    else:
        rtol, atol = 1e-3, 1e-3
    torch.testing.assert_close(tri_out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "shape", [(16, 256), (512, 64), (1023, 64), (1024, 64)]
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_remainder_tt(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = torch.rand(shape, dtype=dtype, device="cpu") * 5 + 1

    ref_out = torch.remainder(x, y)
    tri_out = remainder(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


# =============================================================================
# New tests: div_mode_ (in-place)
# =============================================================================


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512, 64), (1023, 64), (1024, 64)],
)
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_div_mode_inplace_tt(shape, rounding_mode, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = torch.rand(shape, dtype=dtype, device="cpu") * 5 + 1

    ref_out = torch.div(x, y, rounding_mode=rounding_mode)
    x_copy = x.clone()
    div_mode_(x_copy, y, rounding_mode=rounding_mode)

    if dtype == torch.float16 and rounding_mode == "trunc":
        rtol, atol = 1e-2, 1.0
    else:
        rtol, atol = 1e-3, 1e-3
    torch.testing.assert_close(x_copy, ref_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_div_mode_inplace_ts(shape, rounding_mode, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = 5.0

    ref_out = torch.div(x, y, rounding_mode=rounding_mode)
    x_copy = x.clone()
    div_mode_(x_copy, y, rounding_mode=rounding_mode)

    if dtype == torch.float16 and rounding_mode == "trunc":
        rtol, atol = 1e-2, 1.0
    else:
        rtol, atol = 1e-3, 1e-3
    torch.testing.assert_close(x_copy, ref_out, rtol=rtol, atol=atol)


# =============================================================================
# New tests: floor_divide
# =============================================================================


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512, 64), (1023, 64), (1024, 64)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_floor_divide_tt(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = torch.rand(shape, dtype=dtype, device="cpu") * 5 + 1

    ref_out = torch.floor_divide(x, y)
    tri_out = floor_divide(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_floor_divide_ts(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = 5.0

    ref_out = torch.floor_divide(x, y)
    tri_out = floor_divide(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_floor_divide_st(shape, dtype):
    torch.manual_seed(0)
    x = 5.0
    y = torch.rand(shape, dtype=dtype, device="cpu") * 5 + 1

    ref_out = torch.floor_divide(x, y)
    tri_out = floor_divide(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512, 64)],
)
def test_floor_divide_int_tt(shape):
    torch.manual_seed(0)
    x = torch.randint(1, 20, shape, dtype=torch.int32, device="cpu")
    y = torch.randint(1, 10, shape, dtype=torch.int32, device="cpu")

    ref_out = torch.floor_divide(x, y)
    tri_out = floor_divide(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512,)],
)
def test_floor_divide_int_ts(shape):
    torch.manual_seed(0)
    x = torch.randint(1, 20, shape, dtype=torch.int32, device="cpu")
    y = 3

    ref_out = torch.floor_divide(x, y)
    tri_out = floor_divide(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


# =============================================================================
# New tests: floor_divide_ (in-place)
# =============================================================================


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512, 64), (1023, 64), (1024, 64)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_floor_divide_inplace_tt(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = torch.rand(shape, dtype=dtype, device="cpu") * 5 + 1

    ref_out = torch.floor_divide(x, y)
    x_copy = x.clone()
    floor_divide_(x_copy, y)

    torch.testing.assert_close(x_copy, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_floor_divide_inplace_ts(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = 5.0

    ref_out = torch.floor_divide(x, y)
    x_copy = x.clone()
    floor_divide_(x_copy, y)

    torch.testing.assert_close(x_copy, ref_out, rtol=1e-3, atol=1e-3)


# =============================================================================
# New tests: remainder_ts and remainder_st
# =============================================================================


@pytest.mark.parametrize(
    "shape",
    [(512,), (1023,), (1024,)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_remainder_ts(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = 5.0

    ref_out = torch.remainder(x, y)
    tri_out = remainder(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [(512,), (1023,), (1024,)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_remainder_st(shape, dtype):
    torch.manual_seed(0)
    x = 5.0
    y = torch.rand(shape, dtype=dtype, device="cpu") * 5 + 1

    ref_out = torch.remainder(x, y)
    tri_out = remainder(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


# =============================================================================
# New tests: remainder_ (in-place)
# =============================================================================


@pytest.mark.parametrize(
    "shape", [(16, 256), (512, 64), (1023, 64), (1024, 64)]
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_remainder_inplace_tt(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = torch.rand(shape, dtype=dtype, device="cpu") * 5 + 1

    ref_out = torch.remainder(x, y)
    x_copy = x.clone()
    remainder_(x_copy, y)

    torch.testing.assert_close(x_copy, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_remainder_inplace_ts(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = 5.0

    ref_out = torch.remainder(x, y)
    x_copy = x.clone()
    remainder_(x_copy, y)

    torch.testing.assert_close(x_copy, ref_out, rtol=1e-3, atol=1e-3)


# =============================================================================
# New tests: true_divide
# =============================================================================


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512, 64), (1023, 64), (1024, 64)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_true_divide_tt(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = torch.rand(shape, dtype=dtype, device="cpu") * 5 + 1

    ref_out = torch.true_divide(x, y)
    tri_out = true_divide(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_true_divide_ts(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = 5.0

    ref_out = torch.true_divide(x, y)
    tri_out = true_divide(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_true_divide_st(shape, dtype):
    torch.manual_seed(0)
    x = 5.0
    y = torch.rand(shape, dtype=dtype, device="cpu") * 5 + 1

    ref_out = torch.true_divide(x, y)
    tri_out = true_divide(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


# =============================================================================
# New tests: true_divide_ (in-place)
# =============================================================================


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512, 64), (1023, 64), (1024, 64)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_true_divide_inplace_tt(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = torch.rand(shape, dtype=dtype, device="cpu") * 5 + 1

    ref_out = torch.true_divide(x, y)
    x_copy = x.clone()
    true_divide_(x_copy, y)

    torch.testing.assert_close(x_copy, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_true_divide_inplace_ts(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = 5.0

    ref_out = torch.true_divide(x, y)
    x_copy = x.clone()
    true_divide_(x_copy, y)

    torch.testing.assert_close(x_copy, ref_out, rtol=1e-3, atol=1e-3)


# =============================================================================
# New tests: true_divide_out
# =============================================================================


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512, 64), (1023, 64)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_true_divide_out_tt(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = torch.rand(shape, dtype=dtype, device="cpu") * 5 + 1
    out = torch.empty(shape, dtype=dtype, device="cpu")

    ref_out = torch.true_divide(x, y)
    true_divide_out(x, y, out=out)

    torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [(512,), (1023,), (1024,)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_true_divide_out_ts(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 10 + 1
    y = 5.0
    out = torch.empty(shape, dtype=dtype, device="cpu")

    ref_out = torch.true_divide(x, y)
    true_divide_out(x, y, out=out)

    torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [(512,), (1023,), (1024,)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_true_divide_out_st(shape, dtype):
    torch.manual_seed(0)
    x = 5.0
    y = torch.rand(shape, dtype=dtype, device="cpu") * 5 + 1
    out = torch.empty(shape, dtype=dtype, device="cpu")

    ref_out = torch.true_divide(x, y)
    true_divide_out(x, y, out=out)

    torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)
