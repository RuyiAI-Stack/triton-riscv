import importlib

import pytest
import torch

from .special_logsumexp import special_logsumexp


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_special_logsumexp(dim):
    torch.manual_seed(0)
    x = torch.randn(8, 512, dtype=torch.float32) * 4.0

    ref = torch.logsumexp(x, dim=dim)
    tri = special_logsumexp(x, dim)

    assert tri.shape == ref.shape
    # Tolerance source: logsumexp is max-subtracted exp-sum-log in fp32;
    # the row reduction keeps 1e-6 relative accuracy.
    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-6)


def test_special_logsumexp_keepdim():
    torch.manual_seed(0)
    x = torch.randn(8, 16, 32, dtype=torch.float32)

    ref = torch.logsumexp(x, dim=2, keepdim=True)
    tri = special_logsumexp(x, 2, keepdim=True)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-6)


def test_special_logsumexp_overflow_safety():
    # Max subtraction must keep large logits finite and offset-consistent:
    # logsumexp(x + c) == logsumexp(x) + c.
    x = torch.tensor([[1000.0, 999.0, 998.0], [-1000.0, -1001.0, -999.0]])

    ref = torch.logsumexp(x, dim=1)
    tri = special_logsumexp(x, 1)

    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-4)
    shifted = special_logsumexp(x + 5.0, 1)
    torch.testing.assert_close(shifted, tri + 5.0, rtol=1e-6, atol=1e-4)


def test_special_logsumexp_all_neg_inf_row():
    # An all -inf slice reduces to -inf (torch semantics probed on CPU).
    x = torch.tensor([[float("-inf"), float("-inf")], [0.0, 1.0]])

    ref = torch.logsumexp(x, dim=1)
    tri = special_logsumexp(x, 1)

    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-7, equal_nan=True)
    assert tri[0].item() == float("-inf")


def test_special_logsumexp_zero_width_rows():
    # torch reduces an empty axis to -inf per surviving row (probed).
    tri = special_logsumexp(torch.empty(3, 0), 1)
    assert tri.tolist() == [float("-inf")] * 3
    assert special_logsumexp(torch.empty(0, 0), 1).shape == torch.Size([0])


def test_special_logsumexp_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_logsumexp")
    launches = []
    real_run = mod.logsumexp_kernel_inner.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.logsumexp_kernel_inner, "run", spy_run)
    x = torch.randn(8, 128, dtype=torch.float32)
    out = mod.special_logsumexp(x, 1)
    assert len(launches) >= 1, "special_logsumexp must launch a Triton kernel"
    torch.testing.assert_close(out, torch.logsumexp(x, dim=1), rtol=1e-6, atol=1e-6)
