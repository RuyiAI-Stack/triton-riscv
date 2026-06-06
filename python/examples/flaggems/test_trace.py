import pytest
import torch

from .trace import trace


def _trace_reference(x):
    diag = x.diagonal()
    if x.dtype.is_floating_point:
        return diag.sum().to(x.dtype)
    return diag.to(torch.int64).sum()


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
def test_trace(size, dtype):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn((size, size), dtype=dtype, device="cpu")
    else:
        x = torch.randint(0, 100, (size, size), dtype=dtype, device="cpu")

    if dtype == torch.float16:
        # torch.trace is not implemented for CPU float16 in this runtime.
        ref_out = _trace_reference(x)
    else:
        ref_out = torch.trace(x)
    tri_out = trace(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_trace_non_square(dtype):
    torch.manual_seed(0)
    x = torch.randn((512, 256), dtype=dtype, device="cpu")

    ref_out = torch.trace(x)
    tri_out = trace(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_trace_empty(dtype):
    x = torch.randn((0, 512), dtype=dtype, device="cpu")
    ref_out = torch.trace(x)
    tri_out = trace(x)
    torch.testing.assert_close(tri_out, ref_out)
