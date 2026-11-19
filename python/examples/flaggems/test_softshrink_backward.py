import importlib

import pytest
import torch

from .softshrink_backward import softshrink_backward


@pytest.mark.parametrize("size", [512, 1023])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_softshrink_backward_matches_aten(size, dtype):
    torch.manual_seed(0)
    grad = torch.randn(size, dtype=dtype)
    self_t = torch.randn(size, dtype=dtype) * 2.0

    ref = torch.ops.aten.softshrink_backward(grad, self_t, 0.5)
    tri = softshrink_backward(grad, self_t)

    assert tri.dtype == ref.dtype
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6)


def test_softshrink_backward_boundary_matches_hardshrink_mask():
    # Probed on torch CPU: the boundary |x| == lambd is ZEROED (mask is the
    # open interval |x| > lambd), same as hardshrink_backward.
    g = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x = torch.tensor([0.5, -0.5, 0.6, -0.6])

    ref = torch.ops.aten.softshrink_backward(g, x, 0.5)
    tri = softshrink_backward(g, x, 0.5)

    torch.testing.assert_close(tri, ref, rtol=0.0, atol=0.0)
    assert tri[:2].abs().sum().item() == 0.0
    assert tri[2].item() == 3.0 and tri[3].item() == 4.0


def test_softshrink_backward_empty_and_noncontiguous():
    empty = torch.empty(0, dtype=torch.float32)
    assert softshrink_backward(empty, empty, 0.5).shape == (0,)

    g = torch.randn(4, 8)
    x = torch.randn(8, 4).t()
    ref = torch.ops.aten.softshrink_backward(g, x, 0.5)
    torch.testing.assert_close(softshrink_backward(g, x, 0.5), ref, rtol=0.0, atol=1e-6)


def test_softshrink_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.softshrink_backward")
    launches = []
    real_run = mod._softshrink_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._softshrink_backward_kernel, "run", spy_run)
    g = torch.randn(1024)
    x = torch.randn(1024) * 2.0
    out = mod.softshrink_backward(g, x, 0.5)
    assert len(launches) >= 1, "softshrink_backward must launch the Triton kernel"
    torch.testing.assert_close(
        out, torch.ops.aten.softshrink_backward(g, x, 0.5), rtol=0.0, atol=1e-6
    )
