import importlib

import pytest
import torch

from .multilabel_margin_loss_forward import multilabel_margin_loss_forward


@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_multilabel_margin_loss_forward_matches_aten(reduction):
    torch.manual_seed(0)
    x = torch.randn(4, 5)
    t = torch.tensor(
        [[1, 3, -1, -1, -1], [0, 2, 1, -1, -1], [2, -1, -1, -1, -1], [0, 4, 2, -1, -1]]
    )

    loss, is_target = multilabel_margin_loss_forward(x, t, reduction)
    ref_loss, ref_ist = torch.ops.aten.multilabel_margin_loss_forward(x, t, reduction)
    assert loss.shape == ref_loss.shape
    torch.testing.assert_close(loss, ref_loss, rtol=0.0, atol=1e-6)
    # is_target is the probed CLASS-indexed membership mask (labels {1,3}
    # light up columns 1 and 3), not a sequence-position mask.
    assert is_target.dtype == ref_ist.dtype
    torch.testing.assert_close(is_target, ref_ist, rtol=0.0, atol=0.0)


def test_multilabel_margin_loss_forward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.multilabel_margin_loss_forward")
    launches = []
    real_run = mod._multilabel_margin_loss_forward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._multilabel_margin_loss_forward_kernel, "run", spy_run)
    x = torch.randn(4, 5)
    t = torch.tensor(
        [[1, 3, -1, -1, -1], [0, 2, 1, -1, -1], [2, -1, -1, -1, -1], [0, 4, 2, -1, -1]]
    )
    loss, is_target = mod.multilabel_margin_loss_forward(x, t, 0)
    assert len(launches) >= 1, "multilabel_margin_loss_forward must launch the kernel"
    ref = torch.nn.functional.multilabel_margin_loss(x, t, reduction="none")
    torch.testing.assert_close(loss, ref, rtol=0.0, atol=1e-6)


def test_multilabel_margin_loss_forward_empty_batch():
    x = torch.randn(0, 5)
    t = torch.empty(0, 5, dtype=torch.long)
    loss, is_target = multilabel_margin_loss_forward(x, t, 0)
    assert loss.shape == (0,) and is_target.shape == (0, 5)
