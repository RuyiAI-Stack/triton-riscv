import importlib

import pytest
import torch

from .hardshrink_backward import hardshrink_backward


def _aten_ref(grad, self_t, lambd):
    return torch.ops.aten.hardshrink_backward(grad, self_t, lambd)


@pytest.mark.parametrize("size", [512, 1023])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hardshrink_backward_matches_aten(size, dtype):
    torch.manual_seed(0)
    grad = torch.randn(size, dtype=dtype)
    self_t = torch.randn(size, dtype=dtype) * 2.0

    ref = _aten_ref(grad, self_t, 0.5)
    tri = hardshrink_backward(grad, self_t)

    # The mask is exact in the oracle's arithmetic; both paths multiply the
    # same grad element, so equality is bit-tight (rtol=0).
    assert tri.dtype == ref.dtype
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6)


def test_hardshrink_backward_boundary_excludes_band_edges():
    # Probed on torch CPU: the boundary |x| == lambd is ZEROED (the mask is
    # the open interval |x| > lambd); only strictly outside the band the
    # gradient passes unchanged.
    g = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x = torch.tensor([0.5, -0.5, 0.4999, -0.4999, 0.6, -0.6])

    ref = _aten_ref(g, x, 0.5)
    tri = hardshrink_backward(g, x)

    torch.testing.assert_close(tri, ref, rtol=0.0, atol=0.0)
    # On the boundary and inside the band the gradient is exactly zero.
    assert tri[:4].abs().sum().item() == 0.0
    # Strictly outside it passes through unchanged.
    assert tri[4].item() == 5.0 and tri[5].item() == 6.0


def test_hardshrink_backward_empty_and_noncontiguous():
    empty = torch.empty(0, dtype=torch.float32)
    out = hardshrink_backward(empty, empty)
    assert out.shape == (0,)

    g = torch.randn(4, 8)
    x = torch.randn(8, 4).t()  # non-contiguous view
    ref = _aten_ref(g, x, 0.5)
    torch.testing.assert_close(hardshrink_backward(g, x), ref, rtol=0.0, atol=1e-6)


def test_hardshrink_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.hardshrink_backward")
    launches = []
    real_run = mod._hardshrink_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._hardshrink_backward_kernel, "run", spy_run)
    g = torch.randn(1024)
    x = torch.randn(1024) * 2.0
    out = mod.hardshrink_backward(g, x)
    assert len(launches) >= 1, "hardshrink_backward must launch the Triton kernel"
    torch.testing.assert_close(out, _aten_ref(g, x, 0.5), rtol=0.0, atol=1e-6)
