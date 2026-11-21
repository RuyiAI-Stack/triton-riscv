import importlib

import pytest
import torch

from .view_copy import view_copy


def test_view_copy_matches_aten():
    x = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    for size in [(6, 4), (24,), (2, 12), (1, 24), (24, 1)]:
        out = view_copy(x, size)
        ref = torch.ops.aten.view_copy(x, size)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=str(size))
        assert out.shape == torch.Size(size)


def test_view_copy_minus_one_inference():
    x = torch.arange(24, dtype=torch.float32)
    out = view_copy(x, (2, -1, 4))
    assert out.shape == (2, 3, 4)
    torch.testing.assert_close(out, x.reshape(2, 3, 4), rtol=0, atol=0)


def test_view_copy_two_minus_ones_raise():
    x = torch.arange(12, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="only one dimension can be -1"):
        view_copy(x, (-1, -1))


def test_view_copy_numel_mismatch_raises():
    x = torch.arange(12, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="cannot reshape"):
        view_copy(x, (5, 3))


def test_view_copy_strided_input_values_follow_layout():
    # A transposed view reinterpreted by view_copy must follow the view's
    # element order (gather by .contiguous()), not the underlying buffer.
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3).t()  # (3, 2)
    out = view_copy(x, (6,))
    torch.testing.assert_close(out, x.reshape(-1), rtol=0, atol=0)


def test_view_copy_empty():
    x = torch.empty(0, dtype=torch.float32)
    out = view_copy(x, (3, 0, 2))
    assert out.shape == (3, 0, 2)


def test_view_copy_is_value_copy():
    x = torch.arange(6, dtype=torch.float32)
    out = view_copy(x, (2, 3))
    out[0, 0] = -1.0
    assert x[0] == 0.0


def test_view_copy_integer_and_bool():
    for dt in (torch.int64, torch.int32):
        x = torch.arange(6, dtype=dt)
        out = view_copy(x, (3, 2))
        torch.testing.assert_close(
            out, torch.ops.aten.view_copy(x, (3, 2)), rtol=0, atol=0
        )
    xb = torch.arange(6) % 2 == 0
    torch.testing.assert_close(
        view_copy(xb, (3, 2)), torch.ops.aten.view_copy(xb, (3, 2)), rtol=0, atol=0
    )


def test_view_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.view_copy")
    launches = []
    real_run = mod._view_copy_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._view_copy_kernel, "run", spy_run)
    x = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    out = view_copy(x, (6, 4))
    assert len(launches) >= 1, "view_copy must launch the Triton kernel"
    torch.testing.assert_close(out, x.reshape(6, 4), rtol=0, atol=0)
