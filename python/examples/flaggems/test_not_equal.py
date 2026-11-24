import pytest
import torch

from .not_equal import not_equal, not_equal_scalar


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_not_equal(shape):
    torch.manual_seed(0)
    x = torch.randint(0, 4, shape, dtype=torch.int32, device="cpu")
    y = torch.randint(0, 4, shape, dtype=torch.int32, device="cpu")

    ref = torch.not_equal(x, y)
    out = not_equal(x, y)

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_not_equal_scalar(shape):
    torch.manual_seed(0)
    x = torch.randint(0, 4, shape, dtype=torch.int32, device="cpu")

    ref = torch.not_equal(x, 2)
    out = not_equal_scalar(x, 2)

    torch.testing.assert_close(out, ref)
