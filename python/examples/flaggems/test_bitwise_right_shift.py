import pytest
import torch

from .bitwise_right_shift import bitwise_right_shift


@pytest.mark.parametrize("shape", [(16, 256), (1024,)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bitwise_right_shift(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    y = torch.randint(0, 4, shape, dtype=dtype, device="cpu")
    ref = torch.bitwise_right_shift(x, y)
    tri = bitwise_right_shift(x, y)
    torch.testing.assert_close(tri, ref)
