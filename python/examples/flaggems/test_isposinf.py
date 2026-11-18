import pytest
import torch

from .isposinf import isposinf, isposinf_out


# rtol=0, atol=0: boolean output compared exactly (repo test_isneginf convention).
@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_isposinf(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    x[0] = float("inf")
    if shape[0] > 1:
        x[1] = float("-inf")
    if shape[0] > 2:
        x[2] = float("nan")

    ref = torch.isposinf(x)
    tri = isposinf(x)

    assert tri.dtype == torch.bool
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_isposinf_boundary_values():
    # torch semantics: only +inf maps to True; -inf, NaN, zeros and finite
    # values (including large ones) map to False.
    x = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.5, -1.5],
        dtype=torch.float32,
        device="cpu",
    )

    ref = torch.isposinf(x)
    tri = isposinf(x)

    assert tri.dtype == torch.bool
    assert tri.tolist() == [True, False, False, False, False, False, False]
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_isposinf_empty():
    x = torch.empty(0, dtype=torch.float32, device="cpu")

    ref = torch.isposinf(x)
    tri = isposinf(x)

    assert tri.shape == (0,)
    assert tri.dtype == torch.bool
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_isposinf_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, dtype=torch.float32, device="cpu").t()
    assert not x.is_contiguous()
    x[0, 0] = float("inf")
    x[1, 2] = float("-inf")

    ref = torch.isposinf(x)
    tri = isposinf(x)

    assert tri.shape == x.shape
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_isposinf_out(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    x[0] = float("inf")
    x[1] = float("-inf")

    ref = torch.isposinf(x)

    out = torch.empty(shape, dtype=torch.bool, device="cpu")
    ret = isposinf_out(x, out=out)

    assert ret is out
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
