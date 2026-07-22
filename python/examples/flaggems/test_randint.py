import pytest
import torch

from .randint import randint


@pytest.mark.parametrize("high", [1, 7, 97])
def test_randint_matches_torch_distribution(high):
    torch.manual_seed(0)
    out = randint(high, (32768,), dtype=torch.int64, device="cpu")
    torch.manual_seed(0)
    ref = torch.randint(high, (32768,), dtype=torch.int64, device="cpu")

    assert out.shape == ref.shape
    assert out.dtype == torch.int64
    assert torch.all(out >= 0)
    assert torch.all(out < high)
    torch.testing.assert_close(out.float().mean(), ref.float().mean(), rtol=0, atol=0.5)
    torch.testing.assert_close(
        torch.bincount(out, minlength=high).float() / out.numel(),
        torch.bincount(ref, minlength=high).float() / ref.numel(),
        rtol=0,
        atol=2e-2,
    )


def test_randint_out():
    out = torch.empty(1023, dtype=torch.int32, device="cpu")
    result = randint(7, out.shape, dtype=out.dtype, device="cpu", out=out)

    assert result is out
    assert torch.all((0 <= out) & (out < 7))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_randint_empty_matches_torch(dtype):
    out = torch.empty((0, 3), dtype=dtype, device="cpu")

    result = randint(7, out.shape, dtype=dtype, device="cpu", out=out)
    ref = torch.randint(7, out.shape, dtype=dtype, device="cpu")

    assert result is out
    torch.testing.assert_close(out, ref)


def test_randint_rejects_non_positive_high():
    with pytest.raises(RuntimeError):
        torch.randint(0, (1,), device="cpu")
    with pytest.raises(RuntimeError):
        randint(0, (1,), device="cpu")
