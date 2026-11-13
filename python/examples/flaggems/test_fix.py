import pytest
import torch

from .fix import fix, fix_out


def test_fix():
    x = torch.tensor([-1.7, -1.0, 0.0, 1.2, 2.9], dtype=torch.float32, device="cpu")

    out = fix(x)
    ref = torch.fix(x)

    torch.testing.assert_close(out, ref)


def test_fix_out():
    x = torch.tensor([-1.7, 1.2], dtype=torch.float32, device="cpu")
    out = torch.empty_like(x)

    ret = fix_out(x, out)

    assert ret is out
    torch.testing.assert_close(out, torch.fix(x))


def test_fix_preserves_float64_precision():
    x = torch.tensor(
        [float(2**24) - 0.25, -float(2**24) + 0.25],
        dtype=torch.float64,
        device="cpu",
    )

    out = fix(x)
    ref = torch.fix(x)

    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


def test_fix_out_supports_non_contiguous_views():
    x = torch.tensor(
        [[-1.7, 1.2, 0.0], [2.9, -3.1, 4.8]],
        dtype=torch.float32,
        device="cpu",
    ).t()
    out_base = torch.empty((3, 2), dtype=torch.float32, device="cpu")
    out = out_base[:, :]

    ret = fix_out(x, out)

    assert ret is out
    assert not x.is_contiguous()
    torch.testing.assert_close(out, torch.fix(x))


def test_fix_out_resizes_output():
    x = torch.tensor([-1.7, 1.2, 2.9], dtype=torch.float32, device="cpu")
    out = torch.empty(0, dtype=torch.float32, device="cpu")

    returned = fix_out(x, out)

    assert returned is out
    assert out.shape == x.shape
    torch.testing.assert_close(out, torch.fix(x))


def test_fix_out_rejects_dtype_mismatch():
    x = torch.tensor([-1.7, 1.2], dtype=torch.float32, device="cpu")
    out = torch.empty_like(x, dtype=torch.float64)

    with pytest.raises(RuntimeError, match="expected Float"):
        fix_out(x, out)


def test_fix_integer_tensor_is_unchanged():
    x = torch.tensor([-5, 0, 12], dtype=torch.int32, device="cpu")

    out = fix(x)

    assert out.dtype == x.dtype
    torch.testing.assert_close(out, x)


def test_fix_out_rejects_mismatched_dtype():
    x = torch.tensor([1.2], dtype=torch.float32, device="cpu")
    out = torch.empty(1, dtype=torch.int32, device="cpu")

    with pytest.raises(RuntimeError):
        fix_out(x, out)
