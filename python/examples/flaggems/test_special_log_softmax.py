import importlib

import pytest
import torch

from .special_log_softmax import special_log_softmax


@pytest.mark.parametrize("shape, dim", [((2, 512), 1), ((2, 512), -1), ((4, 8, 32), 2)])
def test_special_log_softmax(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32) * 4.0

    ref = torch.log_softmax(x, dim=dim)
    tri = special_log_softmax(x, dim)

    assert tri.dtype == ref.dtype
    # Tolerance source: log_softmax subtracts the rowwise max before exp,
    # so the fp32 result keeps 1e-6 relative accuracy for the row sums; the
    # atol covers log-probabilities near 0.
    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-6)


def test_special_log_softmax_wide_row():
    # Rows wider than one block exercise the large-row kernel path.
    torch.manual_seed(0)
    x = torch.randn(2, 4097, dtype=torch.float32)

    ref = torch.log_softmax(x, dim=1)
    tri = special_log_softmax(x, dim=1)

    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-6)


def test_special_log_softmax_dtype_argument():
    # dtype=float32 upcasts a half input, matching torch.log_softmax
    # (probed on torch CPU: the result dtype follows the dtype argument).
    x = torch.randn(4, 16, dtype=torch.float16)

    ref = torch.log_softmax(x, dim=1, dtype=torch.float32)
    tri = special_log_softmax(x, 1, dtype=torch.float32)

    assert tri.dtype == ref.dtype
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_log_softmax_extremes():
    # Max subtraction must keep very large and very small logits finite;
    # an all -inf row stays -inf (torch semantics).
    x = torch.tensor(
        [[1000.0, 1000.0, -1000.0], [float("-inf"), float("-inf"), float("-inf")]]
    )

    ref = torch.log_softmax(x, dim=1)
    tri = special_log_softmax(x, 1)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)


def test_special_log_softmax_zero_width_rows():
    # a (3, 0) input has three rows of length zero; the port must return
    # the empty shape instead of dividing by an inner extent of 0.
    assert special_log_softmax(torch.empty(3, 0), 1).shape == torch.Size([3, 0])
    assert special_log_softmax(torch.empty(0, 0), 1).shape == torch.Size([0, 0])


def test_special_log_softmax_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_log_softmax")
    launches = []
    real_run = mod.small_n_log_softmax_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.small_n_log_softmax_kernel, "run", spy_run)
    x = torch.randn(8, 128, dtype=torch.float32)
    out = mod.special_log_softmax(x, 1)
    assert len(launches) >= 1, "special_log_softmax must launch a Triton kernel"
    torch.testing.assert_close(out, torch.log_softmax(x, dim=1), rtol=1e-6, atol=1e-6)
