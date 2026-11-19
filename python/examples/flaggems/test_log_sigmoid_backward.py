import importlib

import pytest
import torch

from .log_sigmoid_backward import log_sigmoid_backward


def _contract_buffer(x):
    # Contract established by re-probe: the forward-saved buffer is
    # exp(-|x|), NOT the logsigmoid value itself. Feeding the logsigmoid
    # output instead is a caller error the operator does not validate -- it
    # silently applies where(x<0, 1, buf) / (1 + buf).
    return torch.exp(-x.abs())


def _tolerance(dtype):
    # fp32: the exp/sigmoid composition leaves ~1e-7 on x86 but the
    # aten-vs-libdevice transcendental deviation on riscv64 hosts measured
    # ~1e-6 during earlier ports, so 1e-5 keeps headroom; fp16 runs the fp32
    # compute path (suite band); fp64 is native double (1.5e-16 measured).
    if dtype == torch.float16:
        return 1e-2, 1e-3
    if dtype == torch.float32:
        return 0, 1e-5
    return 0, 1e-6


@pytest.mark.parametrize("size", [512, 1023])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_log_sigmoid_backward_matches_aten_with_contract_buffer(size, dtype):
    torch.manual_seed(0)
    grad = torch.randn(size, dtype=dtype)
    self_t = torch.randn(size, dtype=dtype) * 4.0

    buf = _contract_buffer(self_t)
    ref = torch.ops.aten.log_sigmoid_backward(grad, self_t, buf)
    tri = log_sigmoid_backward(grad, self_t, buf)

    assert tri.dtype == ref.dtype
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


def test_log_sigmoid_backward_math_identity():
    # Independent of the buffer plumbing, the result must equal the closed
    # form at the probed anchors.
    g = torch.ones(6)
    x = torch.tensor([-8.0, -2.0, -0.5, 0.0, 1.0, 2.0])
    tri = log_sigmoid_backward(g, x, _contract_buffer(x))
    math = g * (1 - torch.sigmoid(x))
    torch.testing.assert_close(tri, math, rtol=0.0, atol=1e-5)


def test_log_sigmoid_backward_noncontract_buffer_is_not_validated():
    # Probed on torch CPU: a non-contract buffer raises nothing; the
    # operator applies where(x<0, 1, buf) / (1 + buf). buffer=0 therefore
    # degenerates to the step function x<0 -> 1, x>=0 -> 0 (exact).
    g = torch.ones(4)
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0])
    zeros = torch.zeros_like(x)

    ref = torch.ops.aten.log_sigmoid_backward(g, x, zeros)
    tri = log_sigmoid_backward(g, x, zeros)

    torch.testing.assert_close(tri, ref, rtol=0.0, atol=0.0)
    assert tri[0].item() == 1.0 and tri[1].item() == 1.0
    assert tri[2].item() == 0.0 and tri[3].item() == 0.0


def test_log_sigmoid_backward_empty():
    empty = torch.empty(0)
    assert log_sigmoid_backward(empty, empty, empty).shape == (0,)


def test_log_sigmoid_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.log_sigmoid_backward")
    launches = []
    real_run = mod._log_sigmoid_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._log_sigmoid_backward_kernel, "run", spy_run)
    g = torch.randn(1024)
    x = torch.randn(1024) * 4.0
    buf = _contract_buffer(x)
    out = mod.log_sigmoid_backward(g, x, buf)
    assert len(launches) >= 1, "log_sigmoid_backward must launch the Triton kernel"
    torch.testing.assert_close(
        out,
        torch.ops.aten.log_sigmoid_backward(g, x, buf),
        rtol=0.0,
        atol=1e-5,
    )
