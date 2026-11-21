import importlib

import pytest
import torch

from .unfold_copy import unfold_copy


def test_unfold_copy_2d_dim1_matches_aten():
    x = torch.arange(2 * 8, dtype=torch.float64).reshape(2, 8)
    for size, step in [(3, 1), (3, 2), (8, 3), (1, 4), (5, 5)]:
        out = unfold_copy(x, 1, size, step)
        ref = torch.ops.aten.unfold_copy(x, 1, size, step)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"{size},{step}")
        assert out.shape == ref.shape


def test_unfold_copy_3d_dim2_matches_aten():
    x = torch.arange(2 * 5 * 6, dtype=torch.float64).reshape(2, 5, 6)
    for size, step in [(3, 1), (4, 3), (6, 2), (2, 5)]:
        out = unfold_copy(x, 2, size, step)
        ref = torch.ops.aten.unfold_copy(x, 2, size, step)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"{size},{step}")


def test_unfold_copy_3d_dim1_matches_aten():
    x = torch.arange(2 * 7 * 3, dtype=torch.float64).reshape(2, 7, 3)
    for size, step in [(3, 2), (7, 4), (1, 1), (5, 3)]:
        out = unfold_copy(x, 1, size, step)
        ref = torch.ops.aten.unfold_copy(x, 1, size, step)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"{size},{step}")


def test_unfold_copy_step_zero_raises():
    x = torch.arange(8)
    with pytest.raises(ValueError, match="step must be > 0"):
        unfold_copy(x, 0, 3, 0)


def test_unfold_copy_size_exceeds_dim_raises():
    x = torch.arange(8)
    with pytest.raises(ValueError, match="size must be <= dimension size"):
        unfold_copy(x, 0, 9, 1)


def test_unfold_copy_unsupported_layout_raises():
    # Upstream only implements 2D dim=1 and 3D dim=1/2.
    x1 = torch.arange(8)
    with pytest.raises(NotImplementedError):
        unfold_copy(x1, 0, 3, 1)
    x4 = torch.arange(16).reshape(2, 2, 2, 2)
    with pytest.raises(NotImplementedError):
        unfold_copy(x4, 1, 2, 1)


def test_unfold_copy_dim_wraps_modulo():
    # dimension is taken modulo the rank (upstream behavior, no range check).
    x = torch.arange(2 * 6, dtype=torch.float64).reshape(2, 6)
    out = unfold_copy(x, -1, 3, 1)
    ref = torch.ops.aten.unfold_copy(x, 1, 3, 1)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_unfold_copy_zero_size_windows():
    x = torch.arange(5, dtype=torch.float64)
    # 1D inputs are unsupported upstream, so exercise zero-size via 2D.
    x2 = x.reshape(1, 5)
    out = unfold_copy(x2, 1, 0, 1)
    assert out.shape == (1, 6, 0)


def test_unfold_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.unfold_copy")
    launches = []
    real_run = mod._unfold_copy_kernel_2d.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._unfold_copy_kernel_2d, "run", spy_run)
    x = torch.arange(2 * 8, dtype=torch.float64).reshape(2, 8)
    out = unfold_copy(x, 1, 3, 2)
    assert len(launches) >= 1
    ref = x.unfold(1, 3, 2)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == ref.shape
