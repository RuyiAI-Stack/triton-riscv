import importlib

import pytest
import torch

_UNARY_OPS = [
    "abs", "neg", "ceil", "floor", "trunc", "round", "frac", "sign",
    "reciprocal", "sqrt", "rsqrt", "exp", "expm1", "log", "log10", "log1p",
    "log2", "sin", "cos", "tan", "sinh", "cosh", "tanh", "sigmoid", "erf",
    "erfc", "acos", "asin", "atan", "lgamma",
]

_DOMAINS = {
    "acos": (-0.99, 0.99),
    "asin": (-0.99, 0.99),
    "tan": (-0.6, 0.6),
    "sqrt": (0.1, 3.0),
    "rsqrt": (0.1, 3.0),
    "log": (0.2, 8.0),
    "log2": (0.2, 8.0),
    "log10": (0.2, 8.0),
    "lgamma": (0.2, 8.0),
    "log1p": (-0.5, 2.0),
}

_TOL = {
    "tanh": (1e-4, 1e-6),
    "expm1": (1e-4, 1e-5),
    "erfc": (1e-4, 1e-5),
    "lgamma": (1e-3, 1e-4),
    "sinh": (1e-4, 1e-4),
    "cosh": (1e-4, 1e-4),
}

SHAPES = [(31,), (1024,), (17,)]


def make_inputs(op, seed=0):
    g = torch.Generator().manual_seed(seed)
    lohi = _DOMAINS.get(op)
    if lohi is None:
        return [torch.randn(s, generator=g) for s in SHAPES]
    lo, hi = lohi
    return [torch.rand(s, generator=g) * (hi - lo) + lo for s in SHAPES]


@pytest.mark.parametrize("op", _UNARY_OPS)
def test_foreach_unary_matches_torch(op):
    mod = importlib.import_module("flaggems._foreach_unary")
    # index-based seed: stable across processes (hash() is salted per process)
    xs = make_inputs(op, seed=_UNARY_OPS.index(op) + 1)
    rtol, atol = _TOL.get(op, (1e-5, 1e-6))
    outs = getattr(mod, f"_foreach_{op}")([t.clone() for t in xs])
    for o, x in zip(outs, xs):
        torch.testing.assert_close(o, getattr(torch, op)(x), rtol=rtol, atol=atol)

    holders = [t.clone() for t in xs]
    ret = getattr(mod, f"_foreach_{op}_")(holders)
    assert ret is None
    for h, x in zip(holders, xs):
        torch.testing.assert_close(h, getattr(torch, op)(x), rtol=rtol, atol=atol)


def test_round_half_to_even_exact():
    mod = importlib.import_module("flaggems._foreach_unary")
    x = torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.3, -2.7, 0.0, -0.0, 3.5, -3.5])
    out = mod._foreach_round([x.clone()])
    assert torch.equal(out[0], torch.round(x))


def test_lgamma_reflection_band():
    mod = importlib.import_module("flaggems._foreach_unary")
    x = torch.tensor([0.05, 0.1, 0.2, 0.3, 0.45, 0.49, 1.0, 1.5])
    out = mod._foreach_lgamma([x.clone()])
    torch.testing.assert_close(out[0], torch.lgamma(x), rtol=1e-3, atol=1e-4)


# per-point rtol measured against fp64: <=1.1e-5 for |x|<=2 (1-erf branch),
# <=6.4e-4 approaching the 2.85 seam, <=1.3e-4 above it (asymptotic branch)
_ERFC_POINTS = [
    (0.0, 1e-4), (0.5, 1e-4), (1.0, 1e-4), (1.5, 1e-4), (2.0, 1e-4),
    (2.5, 2e-3), (2.8, 2e-3), (2.85, 2e-3),
    (3.0, 1e-3), (3.5, 1e-3), (4.0, 1e-3), (5.0, 1e-3), (6.0, 1e-3), (8.0, 1e-3),
    (-1.0, 1e-4), (-3.0, 1e-4), (-5.0, 1e-4),
]


def test_erfc_precision_boundary():
    mod = importlib.import_module("flaggems._foreach_unary")
    xs = torch.tensor([p for p, _ in _ERFC_POINTS])
    out = mod._foreach_erfc([xs.clone()])[0]
    ref = torch.erfc(xs.double())
    for (point, rtol), o, r in zip(_ERFC_POINTS, out.tolist(), ref.tolist()):
        assert abs(o - r) / r <= rtol, (point, abs(o - r) / r, rtol)


def test_erfc_non_finite_inputs():
    mod = importlib.import_module("flaggems._foreach_unary")
    x = torch.tensor([float("inf"), float("-inf"), float("nan")])
    out = mod._foreach_erfc([x.clone()])[0]
    assert out[0].item() == 0.0
    assert out[1].item() == 2.0
    assert torch.isnan(out[2])


def test_non_contiguous_inputs():
    mod = importlib.import_module("flaggems._foreach_unary")
    view = torch.randn(16, 8).t()
    out = mod._foreach_neg([view.clone()])
    torch.testing.assert_close(out[0], -view, rtol=0, atol=0)
    base = torch.randn(16, 8)
    orig = base.clone()
    holder = [base.t()]
    mod._foreach_exp_(holder)
    torch.testing.assert_close(base.t(), torch.exp(orig.t()), rtol=1e-5, atol=1e-6)


def test_empty_list():
    mod = importlib.import_module("flaggems._foreach_unary")
    assert mod._foreach_abs([]) == []
    assert mod._foreach_exp_([]) is None


def test_foreach_unary_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._foreach_unary")
    calls = {"n": 0}
    real_run = mod._unary_kernel.run

    def spy_run(*args, **kwargs):
        calls["n"] += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._unary_kernel, "run", spy_run)
    mod._foreach_exp([torch.randn(64)])
    assert calls["n"] == 1
    mod._foreach_abs([torch.randn(64)])
    assert calls["n"] == 2
    mod._foreach_lgamma([torch.rand(64) + 0.5])
    assert calls["n"] == 3


def test_public_surface_complete():
    mod = importlib.import_module("flaggems._foreach_unary")
    for op in _UNARY_OPS:
        assert callable(getattr(mod, f"_foreach_{op}"))
        assert callable(getattr(mod, f"_foreach_{op}_"))
