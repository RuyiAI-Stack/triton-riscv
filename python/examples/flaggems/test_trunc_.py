import torch

from .trunc_ import trunc, trunc_


def test_trunc():
    x = torch.tensor([-2.7, -1.1, 0.0, 1.9, 2.2], dtype=torch.float32, device="cpu")

    out = trunc(x)
    ref = torch.trunc(x)

    torch.testing.assert_close(out, ref)


def test_trunc_inplace():
    x = torch.tensor([-2.7, 1.9], dtype=torch.float32, device="cpu")
    ref = torch.trunc(x)

    ret = trunc_(x)

    assert ret is x
    torch.testing.assert_close(x, ref)


def test_trunc_preserves_float64_precision():
    x = torch.tensor(
        [float(2**24) - 0.25, -float(2**24) + 0.75],
        dtype=torch.float64,
        device="cpu",
    )

    out = trunc(x)
    ref = torch.trunc(x)

    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


def test_trunc_integer_tensor_is_unchanged():
    x = torch.tensor([-3, 0, 9], dtype=torch.int32, device="cpu")

    out = trunc(x)

    assert out.dtype == x.dtype
    torch.testing.assert_close(out, x)


def test_trunc_inplace_supports_noncontiguous_input():
    x = torch.tensor(
        [[-2.7, 1.9, 4.2], [3.8, -0.3, -9.9]],
        dtype=torch.float32,
        device="cpu",
    ).t()
    ref = torch.trunc(x)

    returned = trunc_(x)

    assert returned is x
    assert not x.is_contiguous()
    torch.testing.assert_close(x, ref)
