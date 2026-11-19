import importlib

import pytest
import torch

from .infinitely_differentiable_gelu_backward import (
    infinitely_differentiable_gelu_backward,
)


def _tolerance(dtype):
    # fp32: aten's erf and libdevice's erf were measured to deviate by up to
    # ~8e-6 on riscv64 hosts during the erfinv_ port, so 5e-5 keeps a >6x
    # margin there; fp16 goes through the fp32 compute path and keeps the
    # suite's fp16 band; fp64 is native double (erf measured 1.1e-16).
    if dtype == torch.float16:
        return 1e-2, 1e-3
    if dtype == torch.float32:
        return 0, 5e-5
    return 0, 1e-5


@pytest.mark.parametrize("size", [512, 1023])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_gelu_backward_matches_aten(size, dtype):
    torch.manual_seed(0)
    grad = torch.randn(size, dtype=dtype)
    self_t = torch.randn(size, dtype=dtype) * 3.0

    ref = torch.ops.aten.infinitely_differentiable_gelu_backward(grad, self_t)
    tri = infinitely_differentiable_gelu_backward(grad, self_t)

    assert tri.dtype == ref.dtype
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


def test_gelu_backward_anchor_values():
    # Probed on torch CPU: f'(0) = Phi(0) = 0.5 (the x*phi(x) term is 0).
    g = torch.ones(3)
    x = torch.tensor([0.0, -10.0, 10.0])
    tri = infinitely_differentiable_gelu_backward(g, x)
    ref = torch.ops.aten.infinitely_differentiable_gelu_backward(g, x)
    torch.testing.assert_close(tri, ref, rtol=0, atol=5e-5)
    assert abs(tri[0].item() - 0.5) <= 5e-5
    # Far tails: below -10 the derivative is ~0, above 10 it is ~1.
    assert tri[1].abs().item() <= 1e-4
    assert abs(tri[2].item() - 1.0) <= 1e-4


def test_gelu_backward_empty():
    empty = torch.empty(0)
    assert infinitely_differentiable_gelu_backward(empty, empty).shape == (0,)


def test_gelu_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.infinitely_differentiable_gelu_backward")
    launches = []
    real_run = mod._infinitely_differentiable_gelu_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(
        mod._infinitely_differentiable_gelu_backward_kernel, "run", spy_run
    )
    g = torch.randn(1024)
    x = torch.randn(1024) * 3.0
    out = mod.infinitely_differentiable_gelu_backward(g, x)
    assert len(launches) >= 1, "gelu backward must launch the Triton kernel"
    torch.testing.assert_close(
        out,
        torch.ops.aten.infinitely_differentiable_gelu_backward(g, x),
        rtol=0,
        atol=5e-5,
    )
