import pytest
import torch

from ._is_all_true import _is_all_true


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test__is_all_true_all_true(size):
    torch.manual_seed(0)
    x = torch.ones(size, device="cpu", dtype=torch.bool)

    ref_out = torch.all(x)
    tri_out = _is_all_true(x)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test__is_all_true_some_false(size):
    torch.manual_seed(0)
    x = torch.ones(size, device="cpu", dtype=torch.bool)
    x[size // 2] = False  # Set at least one element to False

    ref_out = torch.all(x)
    tri_out = _is_all_true(x)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test__is_all_true_all_false(size):
    torch.manual_seed(0)
    x = torch.zeros(size, device="cpu", dtype=torch.bool)

    ref_out = torch.all(x)
    tri_out = _is_all_true(x)

    torch.testing.assert_close(tri_out, ref_out)
