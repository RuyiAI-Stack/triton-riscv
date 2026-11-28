import importlib

import pytest
import torch

from .lgamma_ import lgamma, lgamma_


def _tolerance(dtype):
    # Tolerance source: the repo's special_gammaln port (Lanczos g = 7
    # with reflection below 0.5) measured 1e-3/1e-4 in fp32; the fp64
    # path must evaluate in double precision and lands at double ulp,
    # so 1e-9/1e-12 keeps a wide margin.
    if dtype == torch.float64:
        return 1e-9, 1e-12
    return 1e-3, 1e-4


@pytest.mark.parametrize("shape", [(512,), (1023,), (8, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lgamma_matches_aten(shape, dtype):
    # lgamma (functional, same file as lgamma_) is the value oracle
    # companion pinned here alongside the in-place entry.
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype) * 20.0

    ref = torch.lgamma(x)
    tri = lgamma(x)

    assert tri.dtype == ref.dtype
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("shape", [(512,), (1023,), (8, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lgamma_inplace_matches_functional_oracle(shape, dtype):
    torch.manual_seed(0)
    x_orig = torch.rand(shape, dtype=dtype) * 20.0
    x = x_orig.clone()

    ret = lgamma_(x)

    assert ret is x
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(x, torch.lgamma(x_orig), rtol=rtol, atol=atol)
    # self was mutated in place, not rebound to a fresh tensor
    assert not torch.equal(x, x_orig)


def test_lgamma_poles_and_specials():
    # ATen contract probed on torch CPU fp32: poles at 0 and the
    # negative integers give +inf (0.0 and -0.0 both), both infinities
    # give +inf, nan stays nan; lgamma(1) = lgamma(2) = 0 exactly.
    x = torch.tensor(
        [1.0, 2.0, 0.0, -0.0, -1.0, -2.0, float("inf"), float("-inf"), float("nan")]
    )

    ref = torch.lgamma(x)
    tri = lgamma(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == 0.0 and tri[1].item() == 0.0
    assert tri[2].item() == float("inf") and tri[4].item() == float("inf")


def test_lgamma_negative_branch():
    # The reflection formula covers x < 0.5, including the fractional
    # bands between consecutive poles.
    x = torch.tensor([0.1, 0.5, 1.5, -0.1, -0.5, -1.5, -2.5, -10.5, 100.0])

    ref = torch.lgamma(x)
    tri = lgamma(x)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


def test_lgamma_inplace_non_contiguous():
    torch.manual_seed(0)
    base = torch.rand(8, 16) * 20.0
    view = base[::2, ::2]
    assert not view.is_contiguous()
    view_orig = view.clone()

    ret = lgamma_(view)

    assert ret is view
    torch.testing.assert_close(view, torch.lgamma(view_orig), rtol=1e-3, atol=1e-4)
    # the view's storage itself was rewritten
    assert not torch.equal(view, view_orig)


def test_lgamma_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.lgamma_")
    launches = []
    real_run = mod.lgamma_func.run  # kernel name per upstream

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.lgamma_func, "run", spy_run)
    x = torch.rand(1024) * 20.0
    out = lgamma(x)
    assert len(launches) >= 1, "lgamma must launch the Triton kernel"
    torch.testing.assert_close(out, torch.lgamma(x), rtol=1e-3, atol=1e-4)
