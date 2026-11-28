import importlib

import torch

from ._add_relu import _add_relu


def test_add_relu_matches_aten():
    torch.manual_seed(0)
    for size in (512, 1023):
        a = torch.randn(size)
        b = torch.randn(size)
        ref = torch.ops.aten._add_relu(a, b)
        tri = _add_relu(a, b)
        assert tri.dtype == ref.dtype
        torch.testing.assert_close(tri, ref, rtol=0.0, atol=0.0)


def test_add_relu_scalar_mixes():
    a = torch.tensor([-1.0, 0.5, 2.0])
    torch.testing.assert_close(
        _add_relu(a, 1.0), torch.relu(a + 1.0), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        _add_relu(1.0, a), torch.relu(1.0 + a), rtol=0.0, atol=0.0
    )

    both = _add_relu(2.0, 3.0)
    assert torch.is_tensor(both) and both.item() == 5.0
    assert _add_relu(-2.0, 3.0).item() == 1.0
    assert _add_relu(-5.0, 3.0).item() == 0.0


def test_add_relu_empty():
    empty = torch.empty(0)
    assert _add_relu(empty, empty).shape == (0,)


def test_add_relu_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._add_relu")
    launches = []
    real_run = mod._add_relu_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._add_relu_kernel, "run", spy_run)
    a = torch.randn(1024)
    b = torch.randn(1024)
    out = mod._add_relu(a, b)
    assert len(launches) >= 1, "_add_relu must launch the Triton kernel"
    torch.testing.assert_close(out, torch.ops.aten._add_relu(a, b), rtol=0.0, atol=0.0)
