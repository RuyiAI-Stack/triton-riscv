import importlib

import pytest
import torch

from .hardsigmoid_backward import hardsigmoid_backward


@pytest.mark.parametrize("size", [512, 1023])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hardsigmoid_backward_matches_aten(size, dtype):
    torch.manual_seed(0)
    grad = torch.randn(size, dtype=dtype)
    self_t = torch.randn(size, dtype=dtype) * 4.0

    ref = torch.ops.aten.hardsigmoid_backward(grad, self_t)
    tri = hardsigmoid_backward(grad, self_t)

    # g/6 arithmetic: both sides round the same way, keep a tight tolerance
    # for the interpolation constant only.
    assert tri.dtype == ref.dtype
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-7)


def test_hardsigmoid_backward_boundary_saturates_outside_band():
    # Probed on torch CPU: active band is the open interval (-3, 3);
    # x == +-3 is the saturation point and yields zero.
    g = torch.ones(5)
    x = torch.tensor([3.0, -3.0, 2.999, -2.999, 0.0])

    ref = torch.ops.aten.hardsigmoid_backward(g, x)
    tri = hardsigmoid_backward(g, x)

    torch.testing.assert_close(tri, ref, rtol=0.0, atol=0.0)
    assert tri[0].item() == 0.0 and tri[1].item() == 0.0
    torch.testing.assert_close(tri[4], torch.tensor(1.0 / 6.0), rtol=0.0, atol=1e-7)


def test_hardsigmoid_backward_empty():
    empty = torch.empty(0)
    assert hardsigmoid_backward(empty, empty).shape == (0,)


def test_hardsigmoid_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.hardsigmoid_backward")
    launches = []
    real_run = mod._hardsigmoid_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._hardsigmoid_backward_kernel, "run", spy_run)
    g = torch.randn(1024)
    x = torch.randn(1024) * 4.0
    out = mod.hardsigmoid_backward(g, x)
    assert len(launches) >= 1, "hardsigmoid_backward must launch the Triton kernel"
    torch.testing.assert_close(
        out, torch.ops.aten.hardsigmoid_backward(g, x), rtol=0.0, atol=1e-7
    )
