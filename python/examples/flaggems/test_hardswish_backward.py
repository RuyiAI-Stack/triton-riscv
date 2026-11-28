import importlib

import pytest
import torch

from .hardswish_backward import hardswish_backward


@pytest.mark.parametrize("size", [512, 1023])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hardswish_backward_matches_aten(size, dtype):
    torch.manual_seed(0)
    grad = torch.randn(size, dtype=dtype)
    self_t = torch.randn(size, dtype=dtype) * 5.0

    ref = torch.ops.aten.hardswish_backward(grad, self_t)
    tri = hardswish_backward(grad, self_t)

    assert tri.dtype == ref.dtype
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6)


def test_hardswish_backward_piecewise_kinks():
    # Probed on torch CPU: x <= -3 -> 0, x >= 3 -> grad, and the interior
    # branch is (2x + 3) / 6 which pins x = 0 at 0.5 and the seam values.
    g = torch.ones(5)
    x = torch.tensor([-3.0, 3.0, 0.0, -3.0001, 3.0001])

    ref = torch.ops.aten.hardswish_backward(g, x)
    tri = hardswish_backward(g, x)

    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-7)
    assert tri[0].item() == 0.0 and tri[1].item() == 1.0
    assert abs(tri[2].item() - 0.5) <= 1e-7
    assert tri[3].item() == 0.0 and tri[4].item() == 1.0


def test_hardswish_backward_empty_and_noncontiguous():
    empty = torch.empty(0)
    assert hardswish_backward(empty, empty).shape == (0,)

    g = torch.randn(4, 8)
    x = torch.randn(8, 4).t()
    ref = torch.ops.aten.hardswish_backward(g, x)
    torch.testing.assert_close(hardswish_backward(g, x), ref, rtol=0.0, atol=1e-6)


def test_hardswish_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.hardswish_backward")
    launches = []
    real_run = mod._hardswish_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._hardswish_backward_kernel, "run", spy_run)
    g = torch.randn(1024)
    x = torch.randn(1024) * 5.0
    out = mod.hardswish_backward(g, x)
    assert len(launches) >= 1, "hardswish_backward must launch the Triton kernel"
    torch.testing.assert_close(
        out, torch.ops.aten.hardswish_backward(g, x), rtol=0.0, atol=1e-6
    )
