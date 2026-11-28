import importlib

import pytest
import torch

from ._standard_gamma_grad import standard_gamma_grad

# oracle: torch._standard_gamma_grad runs on CPU for fp32/fp64 inputs (probed
# on torch 2.15.0+cpu) — the FlagGems kernel ports ATen's exact three-branch
# formula (Taylor x<0.8, Rice saddle-point alpha>8 near/away from mode,
# bivariate rational default), so the direct comparison is tight for fp32.


def _close(out, ref):
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-4)


def test_standard_gamma_grad_default_branch_matches_aten():
    # alpha <= 8, x >= 0.8: the bivariate rational approximation.
    torch.manual_seed(0)
    alpha = torch.rand(512) * 7.5 + 0.5
    x = torch.rand(512) * 9.0 + 0.8
    out = standard_gamma_grad(alpha, x)
    ref = torch._standard_gamma_grad(alpha, x)
    assert out.shape == alpha.shape and out.dtype == torch.float32
    _close(out, ref)


def test_standard_gamma_grad_small_output_branch_matches_aten():
    # x < 0.8: Taylor series branch.
    torch.manual_seed(1)
    alpha = torch.rand(512) * 4.0 + 0.5
    x = torch.rand(512) * 0.7 + 0.01
    out = standard_gamma_grad(alpha, x)
    ref = torch._standard_gamma_grad(alpha, x)
    _close(out, ref)


def test_standard_gamma_grad_near_mode_branch_matches_aten():
    # alpha > 8 with 0.9*alpha <= x <= 1.1*alpha: Rice near-mode sub-case.
    torch.manual_seed(2)
    alpha = torch.rand(512) * 40.0 + 10.0
    x = alpha * (torch.rand(512) * 0.2 + 0.9)
    out = standard_gamma_grad(alpha, x)
    ref = torch._standard_gamma_grad(alpha, x)
    _close(out, ref)


def test_standard_gamma_grad_away_from_mode_branch_matches_aten():
    # alpha > 8 with x strictly outside [0.9*alpha, 1.1*alpha].
    torch.manual_seed(3)
    alpha = torch.rand(512) * 40.0 + 10.0
    scale = torch.rand(512) * 0.35
    factor = torch.where(torch.rand(512) < 0.5, 0.5 + scale, 1.15 + scale)
    x = alpha * factor
    out = standard_gamma_grad(alpha, x)
    ref = torch._standard_gamma_grad(alpha, x)
    _close(out, ref)


def test_standard_gamma_grad_float64_matches_aten():
    torch.manual_seed(4)
    alpha = torch.rand(256, dtype=torch.float64) * 4.0 + 0.5
    x = torch.rand(256, dtype=torch.float64) * 4.0 + 0.5
    out = standard_gamma_grad(alpha, x)
    ref = torch._standard_gamma_grad(alpha, x)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-8)


def test_standard_gamma_grad_scalar_branch_values_match_aten():
    # One pinned point per branch (probed against aten on this torch build):
    # tiny-x, near-mode, away-from-mode and default.
    torch.manual_seed(5)
    for alpha_v, x_v in [
        (0.2, 0.05),
        (3.0, 0.5),
        (12.0, 11.5),
        (12.0, 20.0),
        (2.0, 1.7),
    ]:
        alpha = torch.tensor([alpha_v])
        x = torch.tensor([x_v])
        _close(standard_gamma_grad(alpha, x), torch._standard_gamma_grad(alpha, x))


def test_standard_gamma_grad_mismatched_dtype_raises():
    torch.manual_seed(6)
    alpha = torch.rand(8) + 0.5
    x = torch.rand(8, dtype=torch.float64) + 0.5
    with pytest.raises(RuntimeError):
        torch._standard_gamma_grad(alpha, x)  # aten rejects the same input
    with pytest.raises(RuntimeError, match="same dtype"):
        standard_gamma_grad(alpha, x)


def test_standard_gamma_grad_integer_dtype_raises():
    alpha = torch.ones(8, dtype=torch.int64)
    x = torch.ones(8, dtype=torch.int64)
    with pytest.raises(RuntimeError):
        torch._standard_gamma_grad(alpha, x)  # aten rejects the same input
    with pytest.raises(RuntimeError, match="expected floating point tensors"):
        standard_gamma_grad(alpha, x)


def test_standard_gamma_grad_triton_path(monkeypatch):
    # The op goes through the module-level pointwise entry
    # _standard_gamma_grad_func; spy on it by wrapping the module attribute.
    mod = importlib.import_module("flaggems._standard_gamma_grad")
    calls = []
    real_func = mod._standard_gamma_grad_func

    def spy_func(*args, **kwargs):
        calls.append(1)
        return real_func(*args, **kwargs)

    monkeypatch.setattr(mod, "_standard_gamma_grad_func", spy_func)
    torch.manual_seed(7)
    alpha = torch.rand(64) * 4.0 + 0.5
    x = torch.rand(64) * 4.0 + 0.5
    out = standard_gamma_grad(alpha, x)
    ref = torch._standard_gamma_grad(alpha, x)
    assert len(calls) >= 1
    _close(out, ref)
