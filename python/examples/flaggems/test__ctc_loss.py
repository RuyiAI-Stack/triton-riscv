import importlib

import pytest
import torch

from ._ctc_loss import _ctc_loss


def _case(T=12, N=3, C=7, S=5, blank=0, seed=0):
    torch.manual_seed(seed)
    lp = torch.randn(T, N, C).log_softmax(2)
    labels = [c for c in range(C) if c != blank]
    tg = torch.tensor(labels)[torch.randint(0, len(labels), (N, S))]
    il = torch.full((N,), T, dtype=torch.long)
    tl = torch.randint(1, S + 1, (N,))
    return lp, tg, il, tl


def test_ctc_loss_matches_aten():
    for seed in range(3):
        lp, tg, il, tl = _case(seed=seed)
        ref = torch.nn.functional.ctc_loss(lp, tg, il, tl, blank=0, reduction="none")
        nll, alpha = _ctc_loss(lp, tg, il.tolist(), tl.tolist(), 0, False)
        assert nll.shape == ref.shape and nll.dtype == ref.dtype
        torch.testing.assert_close(nll, ref, rtol=0.0, atol=1e-4)
        assert alpha.shape == (lp.shape[1], lp.shape[0], 2 * tg.shape[1] + 1)


def test_ctc_loss_alpha_finite_subset_agrees():
    # aten fills unreachable lattice states with a FINITE huge-negative
    # sentinel (~-1e30) after t=0 instead of -inf, while this port keeps
    # -inf there; the values inside the reachable cone agree. Pin the cone
    # elementwise and require aten's out-of-cone slots to be hugely negative
    # wherever this port reports -inf.
    lp, tg, il, tl = _case()
    nll, alpha = _ctc_loss(lp, tg, il, tl, 0, False)
    ref_nll, ref_alpha = torch.ops.aten._ctc_loss(
        lp, tg, il.tolist(), tl.tolist(), 0, False
    )
    torch.testing.assert_close(nll, ref_nll, rtol=0.0, atol=1e-4)
    both_finite = torch.isfinite(ref_alpha) & torch.isfinite(alpha)
    torch.testing.assert_close(
        alpha[both_finite], ref_alpha[both_finite], rtol=0.0, atol=1e-3
    )
    ours_inf = torch.isinf(alpha)
    # aten's unreachable slots are uninitialized memory (probed: alpha
    # hashes differ across processes), so they may hold huge values of
    # either sign or NaN; only require "not a plausible log-probability".
    ref_at_inf = ref_alpha[ours_inf]
    assert bool(((ref_at_inf.abs() > 1e29) | ref_at_inf.isnan()).all()), (
        "aten must hold implausible values wherever this port has -inf"
    )


def test_ctc_loss_flat_targets_equal_padded():
    lp, tg, il, tl = _case()
    flat = torch.cat([tg[n][: tl[n]] for n in range(tg.shape[0])])
    a_nll, _ = _ctc_loss(lp, tg, il, tl, 0, False)
    b_nll, _ = _ctc_loss(lp, flat, il, tl, 0, False)
    torch.testing.assert_close(a_nll, b_nll, rtol=0.0, atol=0.0)


def test_ctc_loss_nonzero_blank_and_zero_length_targets():
    lp, tg, il, tl = _case(blank=3)
    ref = torch.nn.functional.ctc_loss(lp, tg, il, tl, blank=3, reduction="none")
    nll, _ = _ctc_loss(lp, tg, il, tl, 3, False)
    torch.testing.assert_close(nll, ref, rtol=0.0, atol=1e-4)

    # Empty targets: the loss is the sum of -logp[blank] over the input.
    lp2, _, il2, _ = _case(T=6, N=2, S=3, seed=7)
    tl0 = torch.zeros(2, dtype=torch.long)
    ref0 = torch.nn.functional.ctc_loss(
        lp2, torch.zeros(2, 0, dtype=torch.long), il2, tl0, reduction="none"
    )
    nll0, _ = _ctc_loss(lp2, torch.zeros(2, 0, dtype=torch.long), il2, tl0, 0, False)
    torch.testing.assert_close(nll0, ref0, rtol=0.0, atol=1e-4)


def test_ctc_loss_impossible_paths_inf_and_zero_infinity():
    # Probed on torch CPU: target_lengths > input_lengths gives inf, and
    # zero_infinity=True zeroes those entries. S must stay >= the bad tl so
    # only the input-length violation is exercised (a tl > S raises in both
    # this port and aten).
    lp, tg, il, tl = _case(T=4, S=8)
    bad_tl = torch.full_like(tl, 6)
    nll, _ = _ctc_loss(lp, tg, il, bad_tl, 0, False)
    assert bool(torch.isinf(nll).all())
    nll_z, _ = _ctc_loss(lp, tg, il, bad_tl, 0, True)
    assert bool((nll_z == 0).all())


def test_ctc_loss_error_contracts():
    lp, tg, il, tl = _case()
    with pytest.raises(RuntimeError):
        _ctc_loss(lp, tg, il, tl, 99, False)  # blank out of label range
    with pytest.raises(RuntimeError):
        _ctc_loss(
            torch.empty(0, 2, 5),
            torch.zeros(2, 0, dtype=torch.long),
            [0, 0],
            [0, 0],
            0,
            False,
        )
    with pytest.raises(RuntimeError):
        _ctc_loss(
            lp, tg.narrow(1, 0, 2), il, tl, 0, False
        )  # targets narrower than max(tl)


def test_ctc_loss_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._ctc_loss")
    launches = []
    real_run = mod._ctc_loss_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._ctc_loss_kernel, "run", spy_run)
    lp, tg, il, tl = _case(T=8)
    nll, alpha = mod._ctc_loss(lp, tg, il, tl, 0, False)
    assert len(launches) >= 1, "_ctc_loss must launch the Triton kernel"
    ref = torch.nn.functional.ctc_loss(lp, tg, il, tl, reduction="none")
    torch.testing.assert_close(nll, ref, rtol=0.0, atol=1e-4)


def test_ctc_loss_short_input_lengths():
    # Per-sample input_lengths < T: the loss extracts the final states at
    # t = il - 1 (a different code path than the full-length cases).
    torch.manual_seed(5)
    T, N, C, S = 12, 2, 6, 4
    lp = torch.randn(T, N, C).log_softmax(2)
    tg = torch.randint(1, C, (N, S))
    il = torch.tensor([12, 7])
    tl = torch.tensor([3, 4])
    ref = torch.nn.functional.ctc_loss(lp, tg, il, tl, reduction="none")
    nll, alpha = _ctc_loss(lp, tg, il.tolist(), tl.tolist(), 0, False)
    torch.testing.assert_close(nll, ref, rtol=0.0, atol=1e-4)


def test_ctc_loss_repeated_labels_skip_rule():
    # Adjacent duplicate labels forbid the s-2 skip transition; this is the
    # rule that separates a correct lattice from a naive one.
    torch.manual_seed(6)
    T, N, C = 10, 2, 5
    lp = torch.randn(T, N, C).log_softmax(2)
    tg = torch.tensor([[2, 2, 3], [1, 1, 1]])
    il = torch.tensor([10, 10])
    tl = torch.tensor([3, 3])
    ref = torch.nn.functional.ctc_loss(lp, tg, il, tl, reduction="none")
    nll, _ = _ctc_loss(lp, tg, il.tolist(), tl.tolist(), 0, False)
    torch.testing.assert_close(nll, ref, rtol=0.0, atol=1e-4)
