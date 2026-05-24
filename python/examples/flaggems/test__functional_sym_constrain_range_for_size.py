import pytest
import torch

from ._functional_sym_constrain_range_for_size import (
    _functional_sym_constrain_range_for_size,
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test__functional_sym_constrain_range_for_size(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    # Identity function: loads and stores to the same memory
    ref_out = x.clone()

    tri_out = _functional_sym_constrain_range_for_size(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
