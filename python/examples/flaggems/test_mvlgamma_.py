import importlib

import pytest
import torch

from .mvlgamma_ import mvlgamma_


def _tolerance(dtype):
    # Tolerance source: mvlgamma_ sums p lgamma terms; the repo's
    # special_gammaln port measured 1e-3 per term in fp32 and upstream's
    # own suite uses atol=1e-4 vs the fp64 reference, so 1e-3/1e-3 in
    # fp32.  The fp64 path must evaluate the lgamma sum in double
    # precision (aten does), landing at double ulp: 1e-9/1e-12.
    if dtype == torch.float64:
        return 1e-9, 1e-12
    return 1e-3, 1e-3


@pytest.mark.parametrize("p", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(512,), (1023,), (8, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mvlgamma_inplace_matches_functional_oracle(p, shape, dtype):
    # Value oracle via the functional counterpart
    # torch.special.multigammaln (probed equal to legacy torch.mvlgamma
    # on this build; the special spelling avoids the deprecation).
    torch.manual_seed(0)
    x_orig = torch.rand(shape, dtype=dtype) + (p - 1) / 2 + 1.0
    x = x_orig.clone()

    ret = mvlgamma_(x, p)

    assert ret is x
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(
        x, torch.special.multigammaln(x_orig, p), rtol=rtol, atol=atol
    )
    # self was mutated in place, not rebound to a fresh tensor
    assert not torch.equal(x, x_orig)


@pytest.mark.parametrize(
    "p,x",
    [
        (1, 0.0),  # lgamma(0) pole
        (2, -1.0),  # lgamma(-1) pole in the k=0/1 sum
        (3, -2.5),  # lgamma(-3) pole in the k=5 term
        (3, 1.0),  # x = (p - 1) / 2 exactly: lgamma(0) term
        (2, float("inf")),
    ],
)
def test_mvlgamma_inplace_poles_match_aten(p, x):
    # ATen contract probed on torch CPU fp32: any lgamma pole among the
    # summed terms yields +inf; +inf input stays +inf.
    t = torch.tensor([x])

    ret = mvlgamma_(t, p)

    assert ret is t
    ref = torch.special.multigammaln(torch.tensor([x]), p)
    torch.testing.assert_close(t, ref, rtol=0, atol=0, equal_nan=True)
    assert t[0].item() == float("inf")


def test_mvlgamma_inplace_negative_and_special_values():
    # ATen contract probed on torch CPU fp32: negative x away from the
    # half-integer poles is finite; nan propagates.
    xs = torch.tensor([-0.5, -3.2, float("nan")])
    xs_orig = xs.clone()

    ret = mvlgamma_(xs, 2)

    assert ret is xs
    ref = torch.special.multigammaln(xs_orig, 2)
    torch.testing.assert_close(xs, ref, rtol=1e-3, atol=1e-3, equal_nan=True)


def test_mvlgamma_inplace_p_below_one_rejected_like_aten():
    # Exact aten error, probed on torch CPU 2.15: RuntimeError with
    # "p has to be greater than or equal to 1".  (Upstream's KernelGen
    # wrapper raises ValueError("p must be a positive integer"); the
    # repo contract pins aten parity.)
    x = torch.rand(8) * 4.0 + 1.0
    for p in (0, -1):
        with pytest.raises(
            RuntimeError, match="p has to be greater than or equal to 1"
        ):
            mvlgamma_(x, p)


def test_mvlgamma_inplace_non_contiguous():
    torch.manual_seed(0)
    base = torch.rand(8, 16) + 2.0
    view = base[::2, ::2]
    assert not view.is_contiguous()
    view_orig = view.clone()

    ret = mvlgamma_(view, 3)

    assert ret is view
    torch.testing.assert_close(
        view, torch.special.multigammaln(view_orig, 3), rtol=1e-3, atol=1e-3
    )
    # the view's storage itself was rewritten
    assert not torch.equal(view, view_orig)


def test_mvlgamma_inplace_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.mvlgamma_")
    launches = []
    real_run = mod.mvlgamma_kernel_.run  # kernel name per upstream

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.mvlgamma_kernel_, "run", spy_run)
    x = torch.rand(1024) + 2.0
    x_orig = x.clone()
    ret = mvlgamma_(x, 3)
    assert len(launches) >= 1, "mvlgamma_ must launch the Triton kernel"
    torch.testing.assert_close(
        ret, torch.special.multigammaln(x_orig, 3), rtol=1e-3, atol=1e-3
    )
