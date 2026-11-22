import importlib

import pytest
import torch

from .igamma_ import igamma_


def _tolerance(dtype):
    # Tolerance source: same construction and bounds as the repo's
    # special_gammainc suite (series below x = a + 1, Lentz continued
    # fraction above, 1e-4/1e-6 across the seam in fp32); the fp64 path
    # runs the branches in double precision and lands at double ulp.
    if dtype == torch.float64:
        return 1e-9, 1e-12
    return 1e-4, 1e-6


@pytest.mark.parametrize("shape", [(512,), (1023,), (8, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_igamma_inplace_matches_functional_oracle(shape, dtype):
    torch.manual_seed(0)
    a_orig = torch.rand(shape, dtype=dtype) * 9.0 + 0.5
    x = torch.rand(shape, dtype=dtype) * 20.0
    a = a_orig.clone()

    ret = igamma_(a, x)

    assert ret is a
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(
        a, torch.special.gammainc(a_orig, x), rtol=rtol, atol=atol
    )
    # self was mutated in place, not rebound to a fresh tensor
    assert not torch.equal(a, a_orig)


def test_igamma_inplace_boundaries_and_specials():
    # ATen contract probed on torch CPU fp32 (matches the functional
    # oracle torch.special.gammainc): P(a, 0) = 0, x < 0 = nan, a < 0 =
    # nan, x = inf and large x saturate at 1, nan propagates.
    a = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, 0.5, 0.5, 1.0, float("nan")])
    x = torch.tensor(
        [0.0, -1.0, float("-inf"), float("nan"), 1.0, float("inf"), 1e10, 1e10, 1.0]
    )
    a_orig = a.clone()

    ret = igamma_(a, x)

    assert ret is a
    torch.testing.assert_close(
        a, torch.special.gammainc(a_orig, x), rtol=0, atol=0, equal_nan=True
    )
    assert a[0].item() == 0.0
    assert a[5].item() == 1.0 and a[6].item() == 1.0


def test_igamma_inplace_non_contiguous():
    torch.manual_seed(0)
    base = torch.rand(8, 16) * 9.0 + 0.5
    view = base[::2, ::2]
    assert not view.is_contiguous()
    x = torch.rand(4, 8) * 20.0
    view_orig = view.clone()

    ret = igamma_(view, x)

    assert ret is view
    torch.testing.assert_close(
        view, torch.special.gammainc(view_orig, x), rtol=1e-4, atol=1e-6
    )
    # the view's storage itself was rewritten
    assert not torch.equal(view, view_orig)


def test_igamma_inplace_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.igamma_")
    launches = []
    real_run = mod.igamma_kernel_.run  # kernel name per upstream

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.igamma_kernel_, "run", spy_run)
    a = torch.rand(1024) * 4.0 + 0.5
    a_orig = a.clone()
    x = torch.rand(1024) * 10.0
    ret = igamma_(a, x)
    assert len(launches) >= 1, "igamma_ must launch the Triton kernel"
    torch.testing.assert_close(
        ret, torch.special.gammainc(a_orig, x), rtol=1e-4, atol=1e-6
    )
