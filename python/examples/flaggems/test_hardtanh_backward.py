import importlib

import pytest
import torch

from .hardtanh_backward import hardtanh_backward


@pytest.mark.parametrize("size", [512, 1023])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hardtanh_backward_matches_aten(size, dtype):
    torch.manual_seed(0)
    grad = torch.randn(size, dtype=dtype)
    self_t = torch.randn(size, dtype=dtype) * 2.0

    ref = torch.ops.aten.hardtanh_backward(grad, self_t, -1.0, 1.0)
    tri = hardtanh_backward(grad, self_t, -1.0, 1.0)

    assert tri.dtype == ref.dtype
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6)


def test_hardtanh_backward_boundary_is_strictly_interior():
    # Probed on torch CPU: the clamp points themselves are zeroed; only the
    # open interval (min_val, max_val) passes the gradient.
    g = torch.ones(5)
    x = torch.tensor([1.0, -1.0, 1.0001, -1.0001, 0.0])

    ref = torch.ops.aten.hardtanh_backward(g, x, -1.0, 1.0)
    tri = hardtanh_backward(g, x, -1.0, 1.0)

    torch.testing.assert_close(tri, ref, rtol=0.0, atol=0.0)
    assert tri[:4].abs().sum().item() == 0.0
    assert tri[4].item() == 1.0


def test_hardtanh_backward_custom_range_and_empty():
    g = torch.randn(7)
    x = torch.randn(7) * 3.0
    ref = torch.ops.aten.hardtanh_backward(g, x, -2.0, 2.0)
    torch.testing.assert_close(
        hardtanh_backward(g, x, -2.0, 2.0), ref, rtol=0.0, atol=1e-6
    )

    empty = torch.empty(0)
    assert hardtanh_backward(empty, empty, -1.0, 1.0).shape == (0,)


def test_hardtanh_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.hardtanh_backward")
    launches = []
    real_run = mod._hardtanh_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._hardtanh_backward_kernel, "run", spy_run)
    g = torch.randn(1024)
    x = torch.randn(1024) * 2.0
    out = mod.hardtanh_backward(g, x, -1.0, 1.0)
    assert len(launches) >= 1, "hardtanh_backward must launch the Triton kernel"
    torch.testing.assert_close(
        out, torch.ops.aten.hardtanh_backward(g, x, -1.0, 1.0), rtol=0.0, atol=1e-6
    )
