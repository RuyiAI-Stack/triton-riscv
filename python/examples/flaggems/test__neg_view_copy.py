import importlib

import pytest
import torch

from ._neg_view_copy import _neg_view_copy, _neg_view_copy_out


def test_neg_view_copy_matches_aten_floats():
    for dt in (torch.float64, torch.float32, torch.float16):
        x = torch.randn(3, 5).to(dt)
        out = _neg_view_copy(x)
        torch.testing.assert_close(
            out, torch.ops.aten._neg_view_copy(x), rtol=0, atol=0
        )


def test_neg_view_copy_signed_zero_preserved():
    x = torch.tensor([0.0, -0.0], dtype=torch.float64)
    out = _neg_view_copy(x)
    assert out[0].item() == -0.0 or str(out[0]) == "-0.0"
    # v * (-1.0) flips the sign bit exactly: -0.0 -> +0.0, +0.0 -> -0.0
    assert torch.copysign(torch.ones(1), out[0]).item() < 0
    assert torch.copysign(torch.ones(1), out[1]).item() > 0


def test_neg_view_copy_integers_wrap():
    x = torch.tensor([1, -1, 127, -128], dtype=torch.int8)
    out = _neg_view_copy(x)
    torch.testing.assert_close(out, -x, rtol=0, atol=0)


def test_neg_view_copy_strided_input():
    x = torch.randn(4, 6).t()
    out = _neg_view_copy(x)
    torch.testing.assert_close(out, torch.ops.aten._neg_view_copy(x), rtol=0, atol=0)


def test_neg_view_copy_higher_rank_strided():
    x = torch.randn(2, 3, 4).permute(1, 2, 0)
    out = _neg_view_copy(x)
    torch.testing.assert_close(out, torch.ops.aten._neg_view_copy(x), rtol=0, atol=0)


def test_neg_view_copy_unsupported_dtype_raises():
    x = torch.tensor([1 + 2j])
    with pytest.raises(RuntimeError, match="not implemented"):
        _neg_view_copy(x)


def test_neg_view_copy_empty():
    x = torch.empty(0, 4)
    out = _neg_view_copy(x)
    assert out.shape == (0, 4)


def test_neg_view_copy_grad_flow():
    x = torch.randn(4, requires_grad=True)
    out = _neg_view_copy(x)
    out.sum().backward()
    torch.testing.assert_close(x.grad, -torch.ones(4), rtol=0, atol=0)


def test_neg_view_copy_out_variant():
    x = torch.randn(2, 3)
    out = torch.empty(2, 3)
    ret = _neg_view_copy_out(x, out=out)
    assert ret is out
    torch.testing.assert_close(out, -x, rtol=0, atol=0)


def test_neg_view_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._neg_view_copy")
    launches = []
    real_run = mod._neg_flat_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._neg_flat_kernel, "run", spy_run)
    x = torch.randn(64, dtype=torch.float64)
    out = _neg_view_copy(x)
    assert len(launches) >= 1
    torch.testing.assert_close(out, -x, rtol=0, atol=0)


def test_neg_view_copy_strided_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._neg_view_copy")
    launches = []
    real_run = mod._neg_strided_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._neg_strided_kernel, "run", spy_run)
    x = torch.randn(4, 8).t()
    out = _neg_view_copy(x)
    assert len(launches) >= 1
    torch.testing.assert_close(out, -x, rtol=0, atol=0)
