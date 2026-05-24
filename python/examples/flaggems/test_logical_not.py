import pytest
import torch

from .logical_not import logical_not


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_logical_not(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 2, shape, dtype=dtype, device="cpu")

    ref = ~x
    tri = logical_not(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
