import importlib

import torch

from .new_ones import new_ones


def test_new_ones_inherits_dtype_and_device():
    self_t = torch.empty(3, dtype=torch.float64)
    out = new_ones(self_t, (2, 3))
    assert out.dtype == torch.float64
    assert out.shape == (2, 3)
    assert bool((out == 1).all())


def test_new_ones_kwarg_overrides():
    self_t = torch.empty(3, dtype=torch.float32)
    out = new_ones(self_t, (4,), dtype=torch.int64)
    assert out.dtype == torch.int64
    assert out.tolist() == [1, 1, 1, 1]
    out2 = new_ones(self_t, (2, 2), dtype=torch.bool)
    assert out2.dtype is torch.bool
    assert bool(out2.all())


def test_new_ones_empty_size():
    self_t = torch.empty(3)
    out = new_ones(self_t, (0, 4))
    assert out.shape == (0, 4)


def test_new_ones_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.new_ones")
    # ones_kernel lives in flaggems.ones; spy it there
    ones_mod = importlib.import_module("flaggems.ones")
    launches = []
    real_run = ones_mod.ones_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(ones_mod.ones_kernel, "run", spy_run)
    out = new_ones(torch.empty(3), (2, 5))
    assert len(launches) >= 1
    assert bool((out == 1).all())
    assert mod.new_ones is not None
