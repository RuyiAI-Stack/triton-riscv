import importlib

import torch

from .fw_primal_copy import _fw_primal_copy, _fw_primal_copy_out


def test_fw_primal_copy_values():
    for dt in (torch.float64, torch.int32):
        x = (torch.arange(12) * 3).to(dt).reshape(3, 4)
        out = _fw_primal_copy(x, 0)  # level unused
        torch.testing.assert_close(out, x, rtol=0, atol=0)
        assert out is not x


def test_fw_primal_copy_strided():
    x = torch.arange(4 * 6, dtype=torch.float64).reshape(4, 6).t()
    out = _fw_primal_copy(x, 0)
    torch.testing.assert_close(out, x.contiguous(), rtol=0, atol=0)


def test_fw_primal_copy_out_variant():
    x = torch.arange(6, dtype=torch.float64)
    out = torch.empty(6, dtype=torch.float64)
    ret = _fw_primal_copy_out(x, 0, out=out)
    assert ret is out
    torch.testing.assert_close(out, x, rtol=0, atol=0)


def test_fw_primal_copy_empty():
    x = torch.empty(0, 3)
    out = _fw_primal_copy(x, 0)
    assert out.shape == (0, 3)


def test_fw_primal_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.fw_primal_copy")
    launches = []
    real_run = mod._copy_flat_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._copy_flat_kernel, "run", spy_run)
    x = torch.arange(24, dtype=torch.float64).reshape(2, 12)
    out = _fw_primal_copy(x, 0)
    assert len(launches) >= 1
    torch.testing.assert_close(out, x, rtol=0, atol=0)


def test_fw_primal_copy_strided_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.fw_primal_copy")
    launches = []
    real_run = mod._copy_strided_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._copy_strided_kernel, "run", spy_run)
    x = torch.arange(4 * 6, dtype=torch.float64).reshape(4, 6).t()
    out = _fw_primal_copy(x, 0)
    assert len(launches) >= 1
    torch.testing.assert_close(out, x.contiguous(), rtol=0, atol=0)
