import importlib

import pytest
import torch

from ._foreach_reduction import (
    _foreach_max,
    _foreach_norm,
    _foreach_zero,
    _foreach_zero_,
)


def make_list(shapes, seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    return [torch.randn(s, dtype=dtype) for s in shapes]


@pytest.mark.parametrize("shapes", [[(128,)], [(31,), (1023,), (1024,)]])
def test_foreach_max_matches_torch(shapes):
    xs = make_list(shapes)
    ref = torch._foreach_max([x.clone() for x in xs])
    got = _foreach_max([x.clone() for x in xs])
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert g.shape == r.shape == torch.Size([])
        torch.testing.assert_close(g, r, rtol=1e-6, atol=1e-7)


def test_foreach_max_rejects_empty_tensor():
    xs = [torch.randn(4), torch.empty(0)]
    with pytest.raises(RuntimeError, match="Expected reduction dim"):
        _foreach_max(xs)
    with pytest.raises(RuntimeError):
        torch._foreach_max(xs)


def test_foreach_max_non_contiguous():
    torch.manual_seed(0)
    xs = [torch.randn(32, 8).t()]
    ref = torch._foreach_max([x.clone() for x in xs])
    got = _foreach_max(xs)
    torch.testing.assert_close(got[0], ref[0], rtol=1e-6, atol=1e-7)


def test_foreach_max_integer_list():
    xs = [
        torch.tensor([3, -7, 2], dtype=torch.int64),
        torch.tensor([-5, -1, -9, 0], dtype=torch.int64),
    ]
    ref = torch._foreach_max([x.clone() for x in xs])
    got = _foreach_max(xs)
    assert got[0].dtype == torch.int64
    for g, r in zip(got, ref):
        torch.testing.assert_close(g, r, rtol=0, atol=0)


@pytest.mark.parametrize("ord", [0, 1, 2, 3, float("inf"), float("-inf")])
def test_foreach_norm_orders(ord):
    xs = make_list([(128,), (1023,)], seed=1)
    ref = torch._foreach_norm([x.clone() for x in xs], ord)
    got = _foreach_norm([x.clone() for x in xs], ord)
    for g, r in zip(got, ref):
        torch.testing.assert_close(g, r, rtol=1e-5, atol=1e-6)


def test_foreach_norm_default_and_dtype():
    xs = make_list([(64,)], seed=2)
    ref = torch._foreach_norm([x.clone() for x in xs])
    got = _foreach_norm([x.clone() for x in xs])
    torch.testing.assert_close(got[0], ref[0], rtol=1e-5, atol=1e-6)
    got64 = _foreach_norm([x.clone() for x in xs], dtype=torch.float64)
    assert got64[0].dtype == torch.float64


def test_foreach_zero_inplace_matches_torch():
    xs = make_list([(128,), (31, 5)], seed=3)
    refs = [x.clone() for x in xs]
    ret = _foreach_zero_(xs)
    assert ret is None
    torch._foreach_zero_(refs)
    for x, r in zip(xs, refs):
        torch.testing.assert_close(x, r, rtol=0, atol=0)
    assert all((x == 0).all() for x in xs)


def test_foreach_zero_functional_aliases_inputs():
    xs = make_list([(16,), (4, 4)], seed=4)
    refs = [x.clone() for x in xs]
    out = _foreach_zero(xs)
    assert out is not xs or len(out) == len(xs)
    for o, x in zip(out, xs):
        assert o is x
        assert (o == 0).all()
    torch._foreach_zero_(refs)
    for x, r in zip(xs, refs):
        torch.testing.assert_close(x, r, rtol=0, atol=0)


def test_foreach_zero_non_contiguous():
    base = torch.randn(16, 8)
    view = base.t()
    _foreach_zero_([view])
    assert (base == 0).all()


def test_foreach_reduction_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._foreach_reduction")
    calls = {"max": 0, "norm": 0, "zero": 0}
    for name, kernel in (
        ("max", mod._max_kernel),
        ("norm", mod._norm_kernel),
        ("zero", mod._zero_kernel),
    ):
        real_run = kernel.run

        def spy_run(*args, _n=name, _r=real_run, **kwargs):
            calls[_n] += 1
            return _r(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
    xs = make_list([(128,)], seed=5)
    _foreach_max([x.clone() for x in xs])
    assert calls["max"] == 1
    _foreach_norm([x.clone() for x in xs], 2)
    assert calls["norm"] == 1
    _foreach_zero_([x.clone() for x in xs])
    assert calls["zero"] == 1
