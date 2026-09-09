import pytest
import torch

from .special_gammainc import special_gammainc, special_gammainc_out


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_gammainc(shape, dtype):
    torch.manual_seed(0)
    a = torch.rand(shape, dtype=dtype, device="cpu") * 3.5 + 0.5
    x = torch.rand(shape, dtype=dtype, device="cpu") * 4.9 + 0.1

    tri_out = special_gammainc(a, x)
    ref_out = torch.special.gammainc(a, x)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_gammainc_out(shape, dtype):
    torch.manual_seed(0)
    a = torch.rand(shape, dtype=dtype, device="cpu") * 3.5 + 0.5
    x = torch.rand(shape, dtype=dtype, device="cpu") * 4.9 + 0.1
    tri_out = torch.empty_like(a)

    ret = special_gammainc_out(a, x, tri_out)
    ref_out = torch.special.gammainc(a, x)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)
    assert ret is tri_out


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_gammainc_noncontiguous_out(shape, dtype):
    torch.manual_seed(0)
    a = torch.rand(shape, dtype=dtype, device="cpu") * 3.5 + 0.5
    x = torch.rand(shape, dtype=dtype, device="cpu") * 4.9 + 0.1

    if len(shape) == 1:
        tri_out = torch.empty((shape[0] * 2,), dtype=dtype, device="cpu")[::2]
    elif len(shape) == 2:
        tri_out = torch.empty(
            (shape[1], shape[0]), dtype=dtype, device="cpu"
        ).transpose(0, 1)
    else:
        tri_out = torch.empty(shape, dtype=dtype, device="cpu")

    ret = special_gammainc_out(a, x, tri_out)
    assert ret is tri_out
    ref_out = torch.special.gammainc(a, x)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_gammainc_int_inputs_and_broadcast(shape, dtype):
    torch.manual_seed(0)
    a = torch.randint(1, 5, shape, dtype=torch.int32, device="cpu")
    x = torch.rand(shape, dtype=dtype, device="cpu") * 4.9 + 0.1
    tri_out = torch.empty(shape, dtype=dtype, device="cpu")
    special_gammainc_out(a, x, tri_out)
    ref_out = torch.special.gammainc(a.float(), x)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_gammainc_broadcast_inputs(dtype):
    torch.manual_seed(0)
    a = torch.rand((17, 1), dtype=dtype, device="cpu") * 3.5 + 0.5
    x = torch.rand((1, 33), dtype=dtype, device="cpu") * 4.9 + 0.1

    tri_out = special_gammainc(a, x)
    ref_out = torch.special.gammainc(a, x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_gammainc_out_broadcast_inputs(dtype):
    torch.manual_seed(0)
    a = torch.randint(1, 5, (17, 1), dtype=torch.int32, device="cpu")
    x = torch.rand((1, 33), dtype=dtype, device="cpu") * 4.9 + 0.1
    tri_out = torch.empty((17, 33), dtype=dtype, device="cpu")

    ret = special_gammainc_out(a, x, tri_out)
    ref_out = torch.special.gammainc(a.float(), x)

    assert ret is tri_out
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("out_shape", [(0,), (1,), (34, 12)])
def test_special_gammainc_out_broadcast_inputs_resizes_wrong_shape(dtype, out_shape):
    torch.manual_seed(0)
    a = torch.randint(1, 5, (17, 1), dtype=torch.int32, device="cpu")
    x = torch.rand((1, 33), dtype=dtype, device="cpu") * 4.9 + 0.1
    tri_out = torch.empty(out_shape, dtype=dtype, device="cpu")

    ret = special_gammainc_out(a, x, tri_out)
    ref_out = torch.special.gammainc(a.float(), x)

    assert ret is tri_out
    assert tri_out.shape == ref_out.shape == (17, 33)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_gammainc_special_values(dtype):
    a = torch.tensor([1.0, float("inf"), float("inf"), -1.0], dtype=dtype, device="cpu")
    x = torch.tensor(
        [float("inf"), 1.0, float("inf"), float("inf")],
        dtype=dtype,
        device="cpu",
    )

    torch.testing.assert_close(
        special_gammainc(a, x), torch.special.gammainc(a, x), equal_nan=True
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_gammainc_empty_broadcast_inputs(dtype):
    a = torch.empty((0, 1), dtype=dtype, device="cpu")
    x = torch.empty((1, 0), dtype=dtype, device="cpu")

    out = special_gammainc(a, x)
    ref = torch.special.gammainc(a, x)

    assert out.shape == ref.shape == (0, 0)
    torch.testing.assert_close(out, ref)
