import importlib

import pytest
import torch

from .multilabel_margin_loss import multilabel_margin_loss

_RED = {0: "none", 1: "mean", 2: "sum"}


def _case():
    torch.manual_seed(0)
    x = torch.randn(5, 6)
    t = torch.tensor(
        [
            [1, 3, -1, -1, -1, -1],
            [0, 2, 1, -1, -1, -1],
            [1, -1, -1, -1, -1, -1],
            [0, 1, 2, 3, 4, 5],
            [2, 2, -1, -1, -1, -1],  # repeated label: membership is a set
        ]
    )
    return x, t


@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_multilabel_margin_loss_matches_aten(reduction):
    x, t = _case()
    ref = torch.nn.functional.multilabel_margin_loss(x, t, reduction=_RED[reduction])
    tri = multilabel_margin_loss(x, t, reduction)
    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6)


def test_multilabel_margin_loss_hinge_formula_pins():
    # Probed formula: sum_{i in I} sum_{j not in I} max(0, 1 - (x[i] - x[j])) / C,
    # with I = {k : target[k] != -1}; a tie between i and j counts at full
    # hinge weight (x = [1, 1, 0.5], I = {0} gives (1 + 0.5) / 3).
    x = torch.tensor([[1.0, 1.0, 0.5]])
    t = torch.tensor([[0, -1, -1]])
    out = multilabel_margin_loss(x, t, 0)
    torch.testing.assert_close(out, torch.tensor([0.5]), rtol=0.0, atol=1e-6)

    x2 = torch.tensor([[1.0, 2.0, 0.5]])
    t2 = torch.tensor([[0, 1, -1]])
    out2 = multilabel_margin_loss(x2, t2, 0)
    torch.testing.assert_close(out2, torch.tensor([0.5 / 3.0]), rtol=0.0, atol=1e-6)


def test_multilabel_margin_loss_all_negative_row_is_zero():
    # Probed on torch CPU: an empty positive set yields 0.
    x = torch.randn(3, 4)
    t = torch.full((3, 4), -1, dtype=torch.long)
    ref = torch.nn.functional.multilabel_margin_loss(x, t, reduction="none")
    tri = multilabel_margin_loss(x, t, 0)
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6)
    assert bool((tri == 0).all())


def test_multilabel_margin_loss_rejects_out_of_range_target():
    x = torch.randn(3, 4)
    with pytest.raises(RuntimeError):
        multilabel_margin_loss(x, torch.tensor([[1, 4, -1, -1]] * 3), 1)


def test_multilabel_margin_loss_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.multilabel_margin_loss")
    launches = []
    real_run = mod._multilabel_margin_loss_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._multilabel_margin_loss_kernel, "run", spy_run)
    x, t = _case()
    out = mod.multilabel_margin_loss(x, t, 1)
    assert len(launches) >= 1, "multilabel_margin_loss must launch the Triton kernel"
    ref = torch.nn.functional.multilabel_margin_loss(x, t, reduction="mean")
    torch.testing.assert_close(out, ref, rtol=0.0, atol=1e-6)


def test_multilabel_margin_loss_ignores_padding_tail():
    # Probed on torch CPU: the sequence ends at the FIRST negative id, so
    # entries after it are padding -- they do not change the loss and are
    # exempt from the out-of-range check (even 99 must not raise).
    torch.manual_seed(0)
    x = torch.randn(3, 5)
    clean = torch.tensor([[1, 3, -1, -1, -1], [0, 2, 1, -1, -1], [2, -1, -1, -1, -1]])
    dirty = torch.tensor([[1, 3, -1, 4, 2], [0, 2, 1, -1, 3], [2, -1, 4, 0, 1]])
    garbage = torch.tensor([[1, 3, -1, 99, 99], [0, 2, 1, -1, 99], [2, -1, -1, 99, 7]])

    a = multilabel_margin_loss(x, clean, 0)
    b = multilabel_margin_loss(x, dirty, 0)
    c = multilabel_margin_loss(x, garbage, 0)  # must not raise
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)
    torch.testing.assert_close(a, c, rtol=0.0, atol=0.0)


def test_multilabel_margin_loss_empty_batch():
    # Only reduction=none is well-defined in aten here; its empty-batch
    # mean/sum read uninitialized memory (probed: wildly varying values),
    # so this port's sane 0 for those is not pinned against aten.
    x = torch.randn(0, 5)
    t = torch.empty(0, 5, dtype=torch.long)
    assert multilabel_margin_loss(x, t, 0).shape == (0,)
