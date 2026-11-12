import pytest
import torch

from .log_sigmoid_forward import log_sigmoid_forward

BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_log_sigmoid_forward(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 8.0

    ref_out, ref_buffer = torch.ops.aten.log_sigmoid_forward(x)
    tri_out, tri_buffer = log_sigmoid_forward(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(tri_buffer, ref_buffer, rtol=1e-5, atol=1e-5)


def test_log_sigmoid_forward_extremes():
    x = torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0, float("nan")])

    ref_out, ref_buffer = torch.ops.aten.log_sigmoid_forward(x)
    tri_out, tri_buffer = log_sigmoid_forward(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-5, atol=1e-5, equal_nan=True)
    torch.testing.assert_close(
        tri_buffer, ref_buffer, rtol=1e-5, atol=1e-5, equal_nan=True
    )


def test_log_sigmoid_forward_fp64_fallback():
    torch.manual_seed(0)
    x = torch.randn(128, device="cpu", dtype=torch.float64) * 8.0

    ref_out, ref_buffer = torch.ops.aten.log_sigmoid_forward(x)
    tri_out, tri_buffer = log_sigmoid_forward(x)

    torch.testing.assert_close(tri_out, ref_out)
    torch.testing.assert_close(tri_buffer, ref_buffer)


@BF16_XFAIL
def test_log_sigmoid_forward_bf16():
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 8.0).to(torch.bfloat16)

    ref_out, ref_buffer = torch.ops.aten.log_sigmoid_forward(x)
    tri_out, tri_buffer = log_sigmoid_forward(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(tri_buffer, ref_buffer, rtol=1e-2, atol=1e-2)


def test_log_sigmoid_forward_int_raises_like_torch():
    with pytest.raises(NotImplementedError):
        log_sigmoid_forward(torch.tensor([1, -2], dtype=torch.int64))
