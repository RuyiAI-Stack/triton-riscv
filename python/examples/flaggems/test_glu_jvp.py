import importlib

import pytest
import torch

from .glu_jvp import glu_jvp


def _oracle(glu, x, dx, dim):
    # Upstream docstring contract: with N = glu.size(dim),
    # a = x[..., 0:N], b = x[..., N:2N] (narrowed along dim), and the
    # tangents split the same way, the pushforward is
    #     out = da * sigmoid(b) + glu * (1 - sigmoid(b)) * db
    n = glu.size(dim)
    b = x.narrow(dim, n, n)
    da = dx.narrow(dim, 0, n)
    db = dx.narrow(dim, n, n)
    sig_b = torch.sigmoid(b)
    return da * sig_b + glu * (1 - sig_b) * db


def _tolerance(dtype):
    # fp32: sigmoid composition plus the riscv64 aten-vs-libdevice exp
    # headroom measured on earlier ports; fp16 runs the fp32 compute path
    # (suite band); fp64 is native double.
    if dtype == torch.float16:
        return 1e-2, 1e-3
    if dtype == torch.float32:
        return 0, 1e-5
    return 0, 1e-6


@pytest.mark.parametrize("shape", [(2, 6), (3, 4, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_glu_jvp_matches_docstring_formula(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype)
    dx = torch.randn(*shape, dtype=dtype)
    glu = torch.nn.functional.glu(x, dim=-1)

    ref = _oracle(glu, x, dx, -1)
    tri = glu_jvp(glu, x, dx, -1)

    assert tri.dtype == ref.dtype and tri.shape == ref.shape
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


def test_glu_jvp_ignores_first_half_of_x():
    # The docstring states the first half of x is never read; only the
    # second half enters through sigmoid(b). Mutating it must not change
    # the result (exact: same kernel, same inputs otherwise).
    torch.manual_seed(0)
    x = torch.randn(2, 6)
    dx = torch.randn(2, 6)
    glu = torch.nn.functional.glu(x, dim=-1)

    base = glu_jvp(glu, x, dx, -1)
    x2 = x.clone()
    x2[:, :3] += 100.0
    torch.testing.assert_close(glu_jvp(glu, x2, dx, -1), base, rtol=0.0, atol=0.0)


def test_glu_jvp_broadcasts_leading_axes():
    # Upstream contract: every axis except dim broadcasts (glu (1, 3)
    # against x (2, 6) yields (2, 3)).
    torch.manual_seed(0)
    x = torch.randn(2, 6)
    dx = torch.randn(2, 6)
    glu = torch.nn.functional.glu(x, dim=-1)[:1]

    ref = _oracle(glu, x, dx, -1)
    tri = glu_jvp(glu, x, dx, -1)
    assert tri.shape == ref.shape == (2, 3)
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-5)


def test_glu_jvp_rejects_bad_dim_and_dtype():
    x = torch.randn(2, 6)
    dx = torch.randn(2, 6)
    glu = torch.nn.functional.glu(x, dim=-1)
    with pytest.raises(IndexError):
        glu_jvp(glu, x, dx, 5)
    with pytest.raises(TypeError):
        glu_jvp(glu.bool(), x.bool(), dx.bool(), -1)


def test_glu_jvp_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.glu_jvp")
    launches = []
    real_run = mod._glu_jvp_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._glu_jvp_kernel, "run", spy_run)
    x = torch.randn(1024, 6)
    dx = torch.randn(1024, 6)
    glu = torch.nn.functional.glu(x, dim=-1)
    out = mod.glu_jvp(glu, x, dx, -1)
    assert len(launches) >= 1, "glu_jvp must launch the Triton kernel"
    torch.testing.assert_close(out, _oracle(glu, x, dx, -1), rtol=0.0, atol=1e-5)


def test_glu_jvp_empty():
    x = torch.empty(0, 6)
    dx = torch.empty(0, 6)
    glu = torch.empty(0, 3)
    out = glu_jvp(glu, x, dx, -1)
    assert out.shape == (0, 3)
