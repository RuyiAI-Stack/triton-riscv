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


@pytest.mark.parametrize("high", [2**40, 2**62 + 1, 2**63 - 1])
def test_randint_wide_int64_range(high):
    generator = torch.Generator().manual_seed(123)
    result = randint(high, (4096,), generator=generator, dtype=torch.int64)
    assert torch.all((result >= 0) & (result < high))
    # All quarters must be reachable; a 32-bit source cannot reach the upper ones.
    bins = torch.bincount(result // (high // 4 + 1), minlength=4)
    assert torch.all((bins > 800) & (bins < 1250))
    assert torch.any(result % 2 == 1)
    repeated = randint(high, result.shape, generator=torch.Generator().manual_seed(123))
    torch.testing.assert_close(result, repeated)
