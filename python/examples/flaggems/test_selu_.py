import pytest
import torch

from .selu_ import selu_


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_selu_inplace(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    x_ref = x.clone()
    x_before = x.clone()

    ref_out = torch.nn.functional.selu(x_ref)
    result = selu_(x)

    # Check in-place semantics: same data pointer
    assert result.data_ptr() == x.data_ptr()

    # Check values match reference
    torch.testing.assert_close(result, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x, ref_out, rtol=1e-4, atol=1e-4)

    # Verify values actually changed (in-place had effect)
    assert not torch.allclose(x, x_before)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_selu_inplace_forward(size):
    """Forward correctness against torch.nn.functional.selu (not in-place)."""
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    x_copy = x.clone()

    ref_out = torch.nn.functional.selu(x_copy)
    result = selu_(x)

    torch.testing.assert_close(result, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x, ref_out, rtol=1e-4, atol=1e-4)
