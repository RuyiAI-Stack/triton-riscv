import importlib

import pytest
import torch

from ._foreach_binary import (
    _foreach_add_List,
    _foreach_add__List,
    _foreach_add_Scalar,
    _foreach_addcmul_Scalar,
    _foreach_addcdiv_ScalarList,
    _foreach_copy,
    _foreach_copy_,
    _foreach_div_List,
    _foreach_lerp_Scalar,
    _foreach_lerp_List,
    _foreach_mul_List,
    _foreach_mul_Scalar,
    _foreach_mul__Scalar,
    _foreach_pow_ScalarAndTensor,
    _foreach_pow_ScalarList,
    _foreach_clamp_max_Scalar,
    _foreach_clamp_min_List,
    _foreach_maximum_List,
    _foreach_minimum_ScalarList,
    _foreach_sub_List,
    registered_wrappers,
)


def make(shapes, seed=0, kind="randn"):
    g = torch.Generator().manual_seed(seed)
    if kind == "randn":
        return [torch.randn(s, generator=g) for s in shapes]
    if kind == "pos":
        return [torch.rand(s, generator=g) * 2.0 + 0.5 for s in shapes]
    if kind == "exp":
        return [torch.rand(s, generator=g) * 2.0 + 0.5 for s in shapes]
    raise ValueError(kind)


SHAPES = [(31,), (1024,), (17,)]

_LIST_FORMULAS = {
    "add": lambda a, b: a + 2.0 * b,
    "sub": lambda a, b: a - 1.5 * b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "clamp_max": torch.minimum,
    "clamp_min": torch.maximum,
    "maximum": torch.maximum,
    "minimum": torch.minimum,
    "pow": lambda a, b: torch.pow(a, b),
}


@pytest.mark.parametrize("name", list(_LIST_FORMULAS))
def test_list_overload_matches_formula(name):
    kind = "pos" if name in ("div", "pow") else "randn"
    xs = make(SHAPES, seed=1, kind=kind)
    if name == "pow":
        g = torch.Generator().manual_seed(7)
        ys = [torch.rand(s, generator=g) * 2.0 + 0.5 for s in SHAPES]
    elif kind == "pos":
        ys = make(SHAPES, seed=2, kind="randn")
    else:
        ys = make(SHAPES, seed=2, kind="randn")
    wrapper = registered_wrappers()[f"_foreach_{name}.List"]
    alpha = {"add": 2.0, "sub": 1.5}.get(name, 1.0)
    outs = wrapper([t.clone() for t in xs], [t.clone() for t in ys], alpha=alpha)
    for o, x, y in zip(outs, xs, ys):
        torch.testing.assert_close(o, _LIST_FORMULAS[name](x, y), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", ["add", "sub", "mul", "div", "clamp_max", "clamp_min", "maximum", "minimum", "pow"])
def test_scalar_overload_matches_formula(name):
    kind = "pos" if name in ("div", "pow") else "randn"
    xs = make(SHAPES, seed=3, kind=kind)
    s = 0.75 if name == "div" else 2.5
    wrapper = registered_wrappers()[f"_foreach_{name}.Scalar"]
    outs = wrapper([t.clone() for t in xs], s)
    ref_fn = {
        "add": lambda t: t + 2.5,
        "sub": lambda t: t - 2.5,
        "mul": lambda t: t * 2.5,
        "div": lambda t: t / 0.75,
        "clamp_max": lambda t: torch.clamp_max(t, 2.5),
        "clamp_min": lambda t: torch.clamp_min(t, 2.5),
        "maximum": lambda t: torch.maximum(t, torch.full_like(t, 2.5)),
        "minimum": lambda t: torch.minimum(t, torch.full_like(t, 2.5)),
        "pow": lambda t: torch.pow(t, 2.5),
    }[name]
    for o, x in zip(outs, xs):
        torch.testing.assert_close(o, ref_fn(x), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", ["add", "sub", "mul", "div", "clamp_max", "minimum", "pow"])
def test_scalarlist_overload_matches_formula(name):
    kind = "pos" if name in ("div", "pow") else "randn"
    xs = make(SHAPES, seed=4, kind=kind)
    ss = [abs(v) + 0.5 if name == "div" else v for v in [0.5, 2.0, -1.25]]
    wrapper = registered_wrappers()[f"_foreach_{name}.ScalarList"]
    outs = wrapper([t.clone() for t in xs], ss)
    for o, x, s in zip(outs, xs, ss):
        ref = {
            "add": x + s,
            "sub": x - s,
            "mul": x * s,
            "div": x / s,
            "clamp_max": torch.clamp_max(x, s),
            "minimum": torch.minimum(x, torch.full_like(x, s)),
            "pow": torch.pow(x, s),
        }[name]
        torch.testing.assert_close(o, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name,alpha", [("add", 2.0), ("mul", 1.0), ("div", 1.0)])
def test_tensor_overload_broadcasts_one_tensor(name, alpha):
    xs = make(SHAPES, seed=5, kind="randn")
    g = torch.Generator().manual_seed(6)
    if name == "div":
        t = torch.rand((), generator=g) + 0.5
    else:
        t = torch.randn((), generator=g)
    wrapper = registered_wrappers()[f"_foreach_{name}.Tensor"]
    outs = wrapper([x.clone() for x in xs], t, alpha=alpha)
    tf = float(t)
    for o, x in zip(outs, xs):
        expected = {"add": x + alpha * tf, "mul": x * tf, "div": x / tf}[name]
        torch.testing.assert_close(o, expected, rtol=1e-5, atol=1e-6)


def test_inplace_mutates_and_aliases():
    xs = make(SHAPES, seed=8)
    orig = [t.clone() for t in xs]
    ys = make(SHAPES, seed=9)
    ret = _foreach_add__List(xs, ys, alpha=2.0)
    assert ret is None
    for x, o, y in zip(xs, orig, ys):
        torch.testing.assert_close(x, o + 2.0 * y, rtol=1e-5, atol=1e-6)


def test_inplace_scalar_and_copy():
    xs = make(SHAPES, seed=10)
    orig = [t.clone() for t in xs]
    assert _foreach_mul__Scalar(xs, 3.0) is None
    for x, o in zip(xs, orig):
        torch.testing.assert_close(x, o * 3.0, rtol=1e-5, atol=1e-6)
    src = make(SHAPES, seed=11)
    out = _foreach_copy(orig, src)
    for o, s in zip(out, src):
        torch.testing.assert_close(o, s, rtol=0, atol=0)
    bases = [torch.randn(64) for _ in range(2)]
    holders = bases
    _foreach_copy_(holders, [torch.full((64,), 2.0), torch.full((64,), -1.0)])
    assert holders is bases
    torch.testing.assert_close(bases[0], torch.full((64,), 2.0), rtol=0, atol=0)
    torch.testing.assert_close(bases[1], torch.full((64,), -1.0), rtol=0, atol=0)


def test_pow_scalar_and_tensor():
    g = torch.Generator().manual_seed(12)
    exps = [torch.randn(s, generator=g) for s in SHAPES]
    outs = _foreach_pow_ScalarAndTensor(2.0, [t.clone() for t in exps])
    ref = torch._foreach_pow(2.0, exps)
    for o, r in zip(outs, ref):
        torch.testing.assert_close(o, r, rtol=1e-5, atol=1e-6)


def test_lerp_tensor_weight():
    xs = make(SHAPES, seed=13)
    ys = make(SHAPES, seed=14)
    ws = make(SHAPES, seed=15)
    outs = _foreach_lerp_List([t.clone() for t in xs], [t.clone() for t in ys], [t.clone() for t in ws])
    ref = torch._foreach_lerp(xs, ys, ws)
    for o, r in zip(outs, ref):
        torch.testing.assert_close(o, r, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("seed", [16])
def test_lerp_scalar_and_scalarlist_weight(seed):
    xs = make(SHAPES, seed=seed)
    ys = make(SHAPES, seed=seed + 1)
    outs = _foreach_lerp_Scalar([t.clone() for t in xs], [t.clone() for t in ys], 0.25)
    for o, x, y in zip(outs, xs, ys):
        torch.testing.assert_close(o, x + 0.25 * (y - x), rtol=1e-5, atol=1e-6)
    ws = [0.1, 0.6, 0.9]
    outs2 = registered_wrappers()["_foreach_lerp.ScalarList"](
        [t.clone() for t in xs], [t.clone() for t in ys], ws
    )
    for o, x, y, w in zip(outs2, xs, ys, ws):
        torch.testing.assert_close(o, x + w * (y - x), rtol=1e-5, atol=1e-6)


def test_addcmul_scalar_value():
    xs = make(SHAPES, seed=17)
    t1 = make(SHAPES, seed=18)
    t2 = make(SHAPES, seed=19)
    outs = _foreach_addcmul_Scalar([t.clone() for t in xs], t1, t2, value=0.5)
    for o, a, b, c in zip(outs, xs, t1, t2):
        torch.testing.assert_close(o, a + 0.5 * b * c, rtol=1e-5, atol=1e-6)


def test_addcmul_scalarlist_and_tensor_value():
    xs = make(SHAPES, seed=20)
    t1 = make(SHAPES, seed=21)
    t2 = make(SHAPES, seed=22)
    vals = [0.5, 2.0, -0.25]
    outs = registered_wrappers()["_foreach_addcmul.ScalarList"](
        [t.clone() for t in xs], t1, t2, vals
    )
    for o, a, b, c, v in zip(outs, xs, t1, t2, vals):
        torch.testing.assert_close(o, a + v * b * c, rtol=1e-5, atol=1e-6)
    vt = torch.tensor(0.75)
    outs_t = registered_wrappers()["_foreach_addcmul.Tensor"](
        [t.clone() for t in xs], t1, t2, vt
    )
    for o, a, b, c in zip(outs_t, xs, t1, t2):
        torch.testing.assert_close(o, a + 0.75 * b * c, rtol=1e-5, atol=1e-6)


def test_addcdiv_scalarlist():
    xs = make(SHAPES, seed=23)
    t1 = make(SHAPES, seed=24)
    g = torch.Generator().manual_seed(25)
    t2 = [torch.rand(s, generator=g) + 0.5 for s in SHAPES]
    vals = [0.5, 2.0, 1.5]
    outs = _foreach_addcdiv_ScalarList([t.clone() for t in xs], t1, t2, vals)
    for o, a, b, c, v in zip(outs, xs, t1, t2, vals):
        torch.testing.assert_close(o, a + v * b / c, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "call,ref",
    [
        (
            lambda xs, ys: _foreach_add_List(xs, ys, alpha=2.0),
            lambda xs, ys: torch._foreach_add(xs, ys, alpha=2.0),
        ),
        (lambda xs, ys: _foreach_mul_List(xs, ys), lambda xs, ys: torch._foreach_mul(xs, ys)),
        (
            lambda xs, ys: _foreach_div_List(xs, ys),
            lambda xs, ys: torch._foreach_div(xs, ys),
        ),
        (lambda xs, ys: _foreach_sub_List(xs, ys), lambda xs, ys: torch._foreach_sub(xs, ys)),
        (
            lambda xs, ys: _foreach_maximum_List(xs, ys),
            lambda xs, ys: torch._foreach_maximum(xs, ys),
        ),
        (
            lambda xs, ys: _foreach_minimum_ScalarList(xs, [0.5, 2.0, -1.0]),
            lambda xs, ys: [torch.minimum(x, torch.full_like(x, s)) for x, s in zip(xs, [0.5, 2.0, -1.0])],
        ),
        (
            lambda xs, ys: _foreach_pow_ScalarList([x.abs() + 0.1 for x in xs], [2.0, 3.0, 0.5]),
            lambda xs, ys: torch._foreach_pow([x.abs() + 0.1 for x in xs], [2.0, 3.0, 0.5]),
        ),
        (
            lambda xs, ys: _foreach_clamp_max_Scalar(xs, 1.0),
            lambda xs, ys: torch._foreach_clamp_max(xs, 1.0),
        ),
        (
            lambda xs, ys: _foreach_clamp_min_List(xs, [-x.max() for x in xs]),
            lambda xs, ys: [torch.clamp_min(x, -x.max()) for x in xs],
        ),
        (
            lambda xs, ys: _foreach_add_Scalar(xs, 3.0),
            lambda xs, ys: torch._foreach_add(xs, 3.0),
        ),
        (
            lambda xs, ys: _foreach_mul_Scalar(xs, -1.5),
            lambda xs, ys: torch._foreach_mul(xs, -1.5),
        ),
    ],
)
def test_torch_foreach_api_equivalence(call, ref):
    xs = make(SHAPES, seed=26)
    ys = make(SHAPES, seed=27)
    outs = call([t.clone() for t in xs], [t.clone() for t in ys])
    refs = ref([t.clone() for t in xs], [t.clone() for t in ys])
    for o, r in zip(outs, refs):
        torch.testing.assert_close(o, r, rtol=1e-5, atol=1e-6)


def test_non_contiguous_inputs():
    base = torch.randn(16, 8)
    view = base.t()  # 8x16 non-contiguous
    out = _foreach_add_Scalar([view.clone()], 2.0)
    torch.testing.assert_close(out[0], view + 2.0, rtol=1e-5, atol=1e-6)
    xs = [torch.randn(32, 4).t() for _ in range(2)]
    ys = [torch.randn(32, 4).t() for _ in range(2)]
    outs = _foreach_add_List(xs, ys)
    for o, x, y in zip(outs, xs, ys):
        torch.testing.assert_close(o, x + y, rtol=1e-5, atol=1e-6)
    holders = [t.clone() for t in xs]
    _foreach_add__List(holders, ys)
    for h, x, y in zip(holders, xs, ys):
        torch.testing.assert_close(h, x + y, rtol=1e-5, atol=1e-6)


def test_registered_wrappers_surface():
    table = registered_wrappers()
    assert len(table) == 81
    for key in (
        "_foreach_add.List",
        "_foreach_add_.Tensor",
        "_foreach_sub.ScalarList",
        "_foreach_mul.Tensor",
        "_foreach_div_.Scalar",
        "_foreach_clamp_max.List",
        "_foreach_clamp_min.ScalarList",
        "_foreach_maximum_.Scalar",
        "_foreach_minimum.List",
        "_foreach_pow.Scalar",
        "_foreach_lerp.List",
        "_foreach_lerp_.ScalarList",
        "_foreach_addcmul.Scalar",
        "_foreach_addcdiv_.Tensor",
        "_foreach_copy.List",
        "_foreach_copy_.List",
        "_foreach_pow.ScalarAndTensor",
    ):
        assert callable(table[key])


def test_foreach_binary_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._foreach_binary")
    calls = {"pair": 0, "scalar": 0, "ternary": 0}
    for name, kernel in (
        ("pair", mod._pair_kernel),
        ("scalar", mod._scalar_kernel),
        ("ternary", mod._ternary_kernel),
    ):
        real_run = kernel.run

        def spy_run(*args, _n=name, _r=real_run, **kwargs):
            calls[_n] += 1
            return _r(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
    xs = make([(64,)], seed=28)
    ys = make([(64,)], seed=29)
    _foreach_mul_List([x.clone() for x in xs], [y.clone() for y in ys])
    assert calls["pair"] == 1
    _foreach_add_Scalar([x.clone() for x in xs], 2.0)
    assert calls["scalar"] == 1
    _foreach_lerp_Scalar([x.clone() for x in xs], [y.clone() for y in ys], 0.5)
    assert calls["ternary"] == 1
    _foreach_copy([x.clone() for x in xs], [y.clone() for y in ys])
    assert calls["pair"] == 2
    _foreach_pow_ScalarAndTensor(2.0, [x.clone() for x in xs])
    assert calls["scalar"] == 2
