import importlib

import pytest
import torch

from .narrow_copy import narrow_copy


def test_narrow_copy_matches_aten():
    x = torch.arange(2 * 5 * 3, dtype=torch.float64).reshape(2, 5, 3)
    for dim, start, length in [(0, 1, 1), (1, 2, 3), (2, 0, 3), (1, 0, 5), (-1, 1, 2)]:
        out = narrow_copy(x, dim, start, length)
        ref = torch.ops.aten.narrow_copy(x, dim, start, length)
        torch.testing.assert_close(
            out, ref, rtol=0, atol=0, msg=f"{dim},{start},{length}"
        )


def test_narrow_copy_negative_start_wraps():
    x = torch.arange(4 * 6, dtype=torch.float64).reshape(4, 6)
    out = narrow_copy(x, 1, -2, 2)  # -2 % 6 = 4 -> rows 4..5
    ref = torch.ops.aten.narrow_copy(x, 1, 4, 2)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_narrow_copy_length_clamps_silently():
    # Upstream behavior: length beyond the remaining dim is clamped, no error
    # (aten narrow would raise). Pinned here on purpose.
    x = torch.arange(3 * 5, dtype=torch.float64).reshape(3, 5)
    out = narrow_copy(x, 1, 3, 10)
    assert out.shape == (3, 2)
    torch.testing.assert_close(out, x[:, 3:], rtol=0, atol=0)


def test_narrow_copy_invalid_dim_raises():
    x = torch.randn(2, 3)
    with pytest.raises(AssertionError, match="Invalid dim"):
        narrow_copy(x, 2, 0, 1)
    with pytest.raises(AssertionError, match="Invalid dim"):
        narrow_copy(x, -3, 0, 1)


def test_narrow_copy_strided_input():
    x = torch.randn(6, 4).t()  # (4, 6) non-contiguous
    assert not x.is_contiguous()
    out = narrow_copy(x, 1, 1, 3)
    ref = torch.ops.aten.narrow_copy(x, 1, 1, 3)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_narrow_copy_empty_length():
    x = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
    out = narrow_copy(x, 1, 2, 0)
    assert out.shape == (2, 0)
    assert out.numel() == 0


def test_narrow_copy_1d():
    x = torch.arange(7, dtype=torch.float64)
    out = narrow_copy(x, 0, 2, 4)
    torch.testing.assert_close(out, x[2:6], rtol=0, atol=0)
    assert out.shape == (4,)


def test_narrow_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.narrow_copy")
    launches = []
    real_run = mod.narrow_copy_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.narrow_copy_kernel, "run", spy_run)
    x = torch.arange(2 * 5 * 3, dtype=torch.float64).reshape(2, 5, 3)
    out = narrow_copy(x, 1, 1, 3)
    assert len(launches) >= 1, "narrow_copy must launch the Triton kernel"
    torch.testing.assert_close(out, x[:, 1:4], rtol=0, atol=0)
    assert out.shape == (2, 3, 3)
