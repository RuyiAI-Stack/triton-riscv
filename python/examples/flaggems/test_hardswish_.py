import pytest
import torch

from .hardswish_ import hardswish_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64]
)
def test_hardswish_inplace(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    x_ref = x.clone()
    x_ref_out = torch.nn.functional.hardswish(x_ref)
    hardswish_(x)
    torch.testing.assert_close(x, x_ref_out, rtol=1e-2, atol=1e-2)
