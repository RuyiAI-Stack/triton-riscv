import importlib

import pytest
import torch

from .glu_backward_jvp import glu_backward_jvp


def _oracle(grad_x, grad_glu, x, dgrad_glu, dx, dim):
    # Upstream docstring contract, S = grad_glu.size(dim):
    #   a = x.narrow(dim, 0, S), b = x.narrow(dim, S, S), same split for dx
    #   gxa = grad_x.narrow(dim, 0, S)  (second half unused)
    #   sig_b = sigmoid(b); glu = a * sig_b; db_neg = db - db * sig_b
    #   dgrad_x_a = dgrad_glu * sig_b + gxa * db_neg
    #   dgrad_x_b = dgrad_x_a * (a - glu) + gxa * (da - da*sig_b - glu*db_neg)
    #   out = cat([dgrad_x_a, dgrad_x_b], dim)
    s = grad_glu.size(dim)
    a = x.narrow(dim, 0, s)
    b = x.narrow(dim, s, s)
    da = dx.narrow(dim, 0, s)
    db = dx.narrow(dim, s, s)
    gxa = grad_x.narrow(dim, 0, s)
    sig_b = torch.sigmoid(b)
    glu = a * sig_b
    db_neg = db - db * sig_b
    dgrad_x_a = dgrad_glu * sig_b + gxa * db_neg
    dgrad_x_b = dgrad_x_a * (a - glu) + gxa * (da - da * sig_b - glu * db_neg)
    return torch.cat([dgrad_x_a, dgrad_x_b], dim)


def _tolerance(dtype):
    # fp32: sigmoid composition plus the riscv64 aten-vs-libdevice exp
    # headroom measured on earlier ports; fp16 runs the fp32 compute path
    # (suite band); fp64 is native double.
    if dtype == torch.float16:
        return 1e-2, 1e-3
    if dtype == torch.float32:
        return 0, 1e-5
    return 0, 1e-6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_glu_backward_jvp_matches_docstring_formula(dtype):
    torch.manual_seed(0)
    x = torch.randn(2, 6, dtype=dtype)
    dx = torch.randn(2, 6, dtype=dtype)
    grad_x = torch.randn(2, 6, dtype=dtype)
    grad_glu = torch.randn(2, 3, dtype=dtype)
    dgrad_glu = torch.randn(2, 3, dtype=dtype)

    ref = _oracle(grad_x, grad_glu, x, dgrad_glu, dx, -1)
    tri = glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, -1)

    assert tri.dtype == ref.dtype and tri.shape == ref.shape
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


def test_glu_backward_jvp_reads_only_first_half_of_grad_x():
    # Source fact: only grad_x.narrow(dim, 0, S) is loaded; the second half
    # is never read (exact: same kernel, same inputs otherwise).
    torch.manual_seed(0)
    x = torch.randn(2, 6)
    dx = torch.randn(2, 6)
    grad_x = torch.randn(2, 6)
    grad_glu = torch.randn(2, 3)
    dgrad_glu = torch.randn(2, 3)

    base = glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, -1)
    grad_x2 = grad_x.clone()
    grad_x2[:, 3:] += 50.0
    torch.testing.assert_close(
        glu_backward_jvp(grad_x2, grad_glu, x, dgrad_glu, dx, -1),
        base,
        rtol=0.0,
        atol=0.0,
    )


def test_glu_backward_jvp_grad_glu_values_are_not_operands():
    # Source fact: grad_glu only supplies the split extent S; it is never
    # loaded, so changing its values cannot change the result (exact).
    torch.manual_seed(0)
    x = torch.randn(2, 6)
    dx = torch.randn(2, 6)
    grad_x = torch.randn(2, 6)
    dgrad_glu = torch.randn(2, 3)

    base = glu_backward_jvp(grad_x, torch.zeros(2, 3), x, dgrad_glu, dx, -1)
    other = glu_backward_jvp(grad_x, torch.full((2, 3), 7.0), x, dgrad_glu, dx, -1)
    torch.testing.assert_close(base, other, rtol=0.0, atol=0.0)


def test_glu_backward_jvp_rejects_bad_inputs():
    x = torch.randn(2, 6)
    dx = torch.randn(2, 6)
    grad_x = torch.randn(2, 6)
    grad_glu = torch.randn(2, 3)
    dgrad_glu = torch.randn(2, 3)
    with pytest.raises(IndexError):
        glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, 7)
    with pytest.raises(RuntimeError):
        glu_backward_jvp(grad_x.bool(), grad_glu, x, dgrad_glu, dx, -1)


def test_glu_backward_jvp_rejects_float8():
    # Mirrors the upstream gate: float8 operands are rejected outright.
    x = torch.randn(2, 6).to(torch.float8_e4m3fn)
    dx = torch.randn(2, 6).to(torch.float8_e4m3fn)
    grad_x = torch.randn(2, 6).to(torch.float8_e4m3fn)
    grad_glu = torch.randn(2, 3).to(torch.float8_e4m3fn)
    dgrad_glu = torch.randn(2, 3).to(torch.float8_e4m3fn)
    with pytest.raises(RuntimeError):
        glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, -1)


def test_glu_backward_jvp_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.glu_backward_jvp")
    launches = []
    real_run = mod._glu_bwd_jvp_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._glu_bwd_jvp_kernel, "run", spy_run)
    torch.manual_seed(0)
    x = torch.randn(512, 6)
    dx = torch.randn(512, 6)
    grad_x = torch.randn(512, 6)
    grad_glu = torch.randn(512, 3)
    dgrad_glu = torch.randn(512, 3)
    out = mod.glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, -1)
    assert len(launches) >= 1, "glu_backward_jvp must launch the Triton kernel"
    ref = _oracle(grad_x, grad_glu, x, dgrad_glu, dx, -1)
    torch.testing.assert_close(out, ref, rtol=0.0, atol=1e-5)


def test_glu_backward_jvp_empty():
    x = torch.empty(0, 6)
    dx = torch.empty(0, 6)
    grad_x = torch.empty(0, 6)
    grad_glu = torch.empty(0, 3)
    dgrad_glu = torch.empty(0, 3)
    out = glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, -1)
    assert out.shape == (0, 6)
