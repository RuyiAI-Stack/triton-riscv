import importlib

import pytest
import torch

from .mvlgamma import mvlgamma


def _tolerance(dtype):
    # Tolerance source: mvlgamma is a sum of p lgamma terms; the repo's
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
def test_mvlgamma_matches_aten(p, shape, dtype):
    # Oracle: torch.special.multigammaln -- probed equal to the legacy
    # torch.mvlgamma on this build (both 2.6949 at x=3, p=3); the
    # special spelling is picked because torch.mvlgamma is deprecated.
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype) + (p - 1) / 2 + 1.0

    ref = torch.special.multigammaln(x, p)
    tri = mvlgamma(x, p)

    assert tri.dtype == ref.dtype
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


def test_mvlgamma_non_square_trailing_dims():
    # Probed on torch CPU 2.15: mvlgamma is ELEMENTWISE -- a (2, 3)
    # input with p=3 raises no error and matches the oracle entrywise
    # (the task brief expected a square check; aten has none).
    torch.manual_seed(0)
    x = torch.rand(2, 3) + 2.0

    ref = torch.special.multigammaln(x, 3)
    tri = mvlgamma(x, 3)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)
    assert tri.shape == ref.shape


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
def test_mvlgamma_poles_match_aten(p, x):
    # ATen contract probed on torch CPU fp32: any lgamma pole among the
    # summed terms yields +inf; +inf input stays +inf.
    tri = mvlgamma(torch.tensor([x]), p)
    ref = torch.special.multigammaln(torch.tensor([x]), p)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == float("inf")


def test_mvlgamma_negative_and_special_values():
    # ATen contract probed on torch CPU fp32: negative x away from the
    # half-integer poles is finite; nan propagates.
    torch.manual_seed(0)
    xs = torch.tensor([-0.5, -3.2, float("nan")])

    for p in (1, 2):
        tri = mvlgamma(xs.clone(), p)
        ref = torch.special.multigammaln(xs, p)
        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3, equal_nan=True)


def test_mvlgamma_p_below_one_rejected_like_aten():
    # Exact aten error, probed on torch CPU 2.15: RuntimeError with
    # "p has to be greater than or equal to 1".  (Upstream's KernelGen
    # wrapper raises ValueError("p must be a positive integer"); the
    # repo contract pins aten parity.)
    x = torch.rand(8) * 4.0 + 1.0
    for p in (0, -1):
        with pytest.raises(
            RuntimeError, match="p has to be greater than or equal to 1"
        ):
            mvlgamma(x, p)


def test_mvlgamma_non_contiguous():
    torch.manual_seed(0)
    base = torch.rand(8, 16) + 2.0
    view = base.t()
    assert not view.is_contiguous()

    ref = torch.special.multigammaln(view, 3)
    tri = mvlgamma(view, 3)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


def test_mvlgamma_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.mvlgamma")
    launches = []
    real_run = mod.mvlgamma_kernel.run  # kernel name per upstream

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.mvlgamma_kernel, "run", spy_run)
    x = torch.rand(1024) + 2.0
    out = mvlgamma(x, 3)
    assert len(launches) >= 1, "mvlgamma must launch the Triton kernel"
    torch.testing.assert_close(
        out, torch.special.multigammaln(x, 3), rtol=1e-3, atol=1e-3
    )
