import pytest
import torch

from .hardshrink import hardshrink, hardshrink_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_hardshrink(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 2.0
    x[::9] = 0.0

    ref_out = torch.nn.functional.hardshrink(x)
    tri_out = hardshrink(x)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("lambd", [0.5, 2.0])
def test_hardshrink_custom_lambd(lambd):
    torch.manual_seed(0)
    x = torch.randn(1024, device="cpu", dtype=torch.float32) * 3.0

    ref_out = torch.nn.functional.hardshrink(x, lambd=lambd)
    tri_out = hardshrink(x, lambd=lambd)

    torch.testing.assert_close(tri_out, ref_out)


def test_hardshrink_out():
    torch.manual_seed(0)
    x = torch.randn(1024, device="cpu", dtype=torch.float32) * 2.0

    ref_out = torch.nn.functional.hardshrink(x)
    out = torch.empty_like(x)
    ret = hardshrink_out(x, out=out)

    assert ret is out
    torch.testing.assert_close(out, ref_out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_hardshrink_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 2.0).to(dtype)

    ref_out = torch.nn.functional.hardshrink(x)
    tri_out = hardshrink(x)

    torch.testing.assert_close(tri_out, ref_out)


def test_hardshrink_bf16():
    # hardshrink stays in the input dtype (zeros built as x * 0), so this is
    # expected to pass despite issue #114.
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 2.0).to(torch.bfloat16)

    ref_out = torch.nn.functional.hardshrink(x)
    tri_out = hardshrink(x)

    torch.testing.assert_close(tri_out, ref_out)


def test_hardshrink_out_rejects_positional_out():
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0])
    out_t = torch.empty(4)
    with pytest.raises(TypeError):
        hardshrink_out(x, 0.5, out_t)


def test_hardshrink_int_raises_like_torch():
    with pytest.raises(NotImplementedError):
        hardshrink(torch.tensor([1, -2], dtype=torch.int64))


def test_hardshrink_out_small_buffer_raises():
    x = torch.randn(32)
    out = torch.empty(16)
    with pytest.raises(ValueError):
        hardshrink_out(x, out=out)
