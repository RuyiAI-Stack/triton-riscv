import importlib

import torch

from .scalar_tensor import scalar_tensor


def test_scalar_tensor_basic():
    out = scalar_tensor(3.5, dtype=torch.float64)
    assert out.shape == ()
    assert out.dtype == torch.float64
    assert out.item() == 3.5
    torch.testing.assert_close(
        out, torch.ops.aten.scalar_tensor(3.5, dtype=torch.float64), rtol=0, atol=0
    )


def test_scalar_tensor_dtypes():
    for dt, v in [
        (torch.float32, 1.25),
        (torch.int64, 7),
        (torch.int32, -3),
        (torch.float16, 0.5),
    ]:
        out = scalar_tensor(v, dtype=dt)
        assert out.dtype == dt
        assert out.item() == v


def test_scalar_tensor_bool_conversion():
    out = scalar_tensor(2, dtype=torch.bool)
    assert out.dtype is torch.bool
    assert out.item() is True
    out0 = scalar_tensor(0, dtype=torch.bool)
    assert out0.item() is False


def test_scalar_tensor_matches_aten():
    for dt in (torch.float32, torch.float64, torch.int64):
        v = 2.5 if dt.is_floating_point else 3
        torch.testing.assert_close(
            scalar_tensor(v, dtype=dt), torch.scalar_tensor(v, dtype=dt), rtol=0, atol=0
        )


def test_scalar_tensor_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.scalar_tensor")
    launches = []
    real_run = mod.scalar_tensor_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.scalar_tensor_kernel, "run", spy_run)
    out = scalar_tensor(9.5, dtype=torch.float64)
    assert len(launches) >= 1
    assert out.item() == 9.5
    assert out.shape == ()
