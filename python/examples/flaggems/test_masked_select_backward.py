import importlib

import pytest
import torch

from .masked_select_backward import masked_select_backward


def _ref(grad, x, mask):
    out = torch.zeros_like(x)
    sel = torch.nonzero(mask.reshape(-1)).reshape(-1)
    flat = out.reshape(-1)
    flat[sel] = grad
    return out


def test_masked_select_backward_single_pass_matches_ref():
    torch.manual_seed(0)
    x = torch.randn(4, 5)
    mask = torch.rand(4, 5) > 0.4
    grad = torch.randn(int(mask.sum()))
    out = masked_select_backward(grad, x, mask)
    torch.testing.assert_close(out, _ref(grad, x, mask), rtol=0, atol=0)
    ref = torch.ops.aten.masked_select_backward(grad, x, mask)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_masked_select_backward_redundant_prefix_tier():
    torch.manual_seed(1)
    n = 100000  # 32768 < n <= 262144 -> row kernel
    x = torch.randn(n)
    mask = torch.rand(n) > 0.3
    grad = torch.randn(int(mask.sum()))
    out = masked_select_backward(grad, x, mask)
    torch.testing.assert_close(out, _ref(grad, x, mask), rtol=0, atol=0)


def test_masked_select_backward_three_kernel_tier():
    torch.manual_seed(2)
    n = 300000  # > 262144 -> count/prefix/scatter
    x = torch.randn(n)
    mask = torch.rand(n) > 0.5
    grad = torch.randn(int(mask.sum()))
    out = masked_select_backward(grad, x, mask)
    torch.testing.assert_close(out, _ref(grad, x, mask), rtol=0, atol=0)


def test_masked_select_backward_all_false_mask():
    x = torch.randn(6)
    mask = torch.zeros(6, dtype=torch.bool)
    grad = torch.randn(0)
    out = masked_select_backward(grad, x, mask)
    torch.testing.assert_close(out, torch.zeros(6), rtol=0, atol=0)


def test_masked_select_backward_all_true_mask():
    x = torch.randn(6)
    mask = torch.ones(6, dtype=torch.bool)
    grad = torch.arange(6, dtype=torch.float64).to(torch.float32)
    out = masked_select_backward(grad, x, mask)
    torch.testing.assert_close(out, grad, rtol=0, atol=0)


def test_masked_select_backward_broadcast_mask():
    x = torch.randn(3, 4)
    mask = torch.tensor([True, False, True, False])  # broadcasts along rows
    grad = torch.randn(6)
    out = masked_select_backward(grad, x, mask)
    torch.testing.assert_close(out, _ref(grad, x, mask.expand(3, 4)), rtol=0, atol=0)


def test_masked_select_backward_complex():
    torch.manual_seed(3)
    x = torch.randn(3, 4, dtype=torch.complex128)
    mask = torch.rand(3, 4) > 0.5
    grad = torch.randn(int(mask.sum()), dtype=torch.complex128)
    out = masked_select_backward(grad, x, mask)
    torch.testing.assert_close(out, _ref(grad, x, mask), rtol=0, atol=0)


def test_masked_select_backward_nonbool_mask_raises():
    x = torch.randn(3)
    mask = torch.ones(3, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="boolean masks"):
        masked_select_backward(torch.randn(3), x, mask)


def test_masked_select_backward_dtype_mismatch_raises():
    x = torch.randn(3)
    mask = torch.ones(3, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="same dtypes"):
        masked_select_backward(torch.randn(3, dtype=torch.float64), x, mask)


def test_masked_select_backward_empty():
    x = torch.empty(0, 3)
    mask = torch.ones(0, 3, dtype=torch.bool)
    grad = torch.randn(0)
    out = masked_select_backward(grad, x, mask)
    assert out.shape == (0, 3)


def test_masked_select_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.masked_select_backward")
    launches = []
    real_run = mod._masked_select_backward_single_pass_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._masked_select_backward_single_pass_kernel, "run", spy_run)
    torch.manual_seed(0)
    x = torch.randn(4, 5)
    mask = torch.rand(4, 5) > 0.4
    grad = torch.randn(int(mask.sum()))
    out = masked_select_backward(grad, x, mask)
    assert len(launches) >= 1
    sel = torch.nonzero(mask.reshape(-1)).reshape(-1)
    expect = torch.zeros(20)
    expect[sel] = grad
    torch.testing.assert_close(out.reshape(-1), expect, rtol=0, atol=0)
