import pytest
import torch

from .trace import trace


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
def test_trace(size, dtype):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn((size, size), dtype=dtype, device="cpu")
    else:
        x = torch.randint(0, 100, (size, size), dtype=dtype, device="cpu")

    ref_out = x.diagonal().sum()
    tri_out = trace(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_trace_non_square(dtype):
    torch.manual_seed(0)
    x = torch.randn((512, 256), dtype=dtype, device="cpu")

    ref_out = x.diagonal().sum()
    tri_out = trace(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_trace_empty(dtype):
    x = torch.randn((0, 512), dtype=dtype, device="cpu")
    ref_out = torch.trace(x)
    tri_out = trace(x)
    torch.testing.assert_close(tri_out, ref_out)
