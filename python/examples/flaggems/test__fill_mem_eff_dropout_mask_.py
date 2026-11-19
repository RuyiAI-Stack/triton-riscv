import importlib
import inspect

import pytest
import torch

from ._fill_mem_eff_dropout_mask_ import _fill_mem_eff_dropout_mask_


def _mask():
    return torch.empty(2, 3, 4, 5, dtype=torch.float32)


def test_fill_mem_eff_returns_self_filled_unit_range():
    m = _mask()
    out = _fill_mem_eff_dropout_mask_(m, 0.5, 123, 0)
    assert out.data_ptr() == m.data_ptr(), "must return the same storage"
    assert bool((m >= 0).all()) and bool((m < 1).all())


def test_fill_mem_eff_deterministic_and_offset_sensitive():
    a = _fill_mem_eff_dropout_mask_(_mask(), 0.5, 42, 0).clone()
    b = _fill_mem_eff_dropout_mask_(_mask(), 0.5, 42, 0).clone()
    assert torch.equal(a, b), "same (seed, offset) must reproduce the stream"

    c = _fill_mem_eff_dropout_mask_(_mask(), 0.5, 42, 4).clone()
    assert not torch.equal(a, c), "a different offset draws a different stream"


def test_fill_mem_eff_empty_tensor_is_passthrough():
    m = torch.empty(0, 3, 4, 5, dtype=torch.float32)
    out = _fill_mem_eff_dropout_mask_(m, 0.5, 123, 0)
    assert out.data_ptr() == m.data_ptr() and out.numel() == 0


def test_fill_mem_eff_seed_selects_the_stream():
    a = _fill_mem_eff_dropout_mask_(_mask(), 0.5, 1, 0).clone()
    b = _fill_mem_eff_dropout_mask_(_mask(), 0.5, 2, 0).clone()
    assert not torch.equal(a, b), "a different seed draws a different stream"


def test_fill_mem_eff_dropout_p_does_not_participate():
    # Mirrors upstream: the operator fills a plain U[0,1) stream and the
    # dropout probability is not wired into the kernel.
    a = _fill_mem_eff_dropout_mask_(_mask(), 0.0, 7, 0).clone()
    b = _fill_mem_eff_dropout_mask_(_mask(), 0.99, 7, 0).clone()
    assert torch.equal(a, b)


def test_fill_mem_eff_rejects_bad_layout():
    with pytest.raises(AssertionError):
        _fill_mem_eff_dropout_mask_(torch.empty(8), 0.5, 1, 0)  # not 4D
    with pytest.raises(AssertionError):
        _fill_mem_eff_dropout_mask_(
            torch.empty(2, 3, 4, 5, dtype=torch.float64), 0.5, 1, 0
        )
    with pytest.raises(AssertionError):
        noncontig = torch.empty(3, 4 * 2, 4, 5)[:, ::2]
        _fill_mem_eff_dropout_mask_(noncontig, 0.5, 1, 0)


def test_fill_mem_eff_docstring_states_dropout_p_inert():
    doc = inspect.getdoc(_fill_mem_eff_dropout_mask_) or ""
    assert "dropout_p" in doc and "does not participate" in doc


def test_fill_mem_eff_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._fill_mem_eff_dropout_mask_")
    launches = []
    real_run = mod._fill_mem_eff_dropout_mask_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._fill_mem_eff_dropout_mask_kernel, "run", spy_run)
    m = _mask()
    mod._fill_mem_eff_dropout_mask_(m, 0.5, 123, 0)
    assert len(launches) >= 1, "the mask kernel must launch"
    assert bool((m >= 0).all()) and bool((m < 1).all()), (
        "the spy path must still fill the unit-range stream"
    )
