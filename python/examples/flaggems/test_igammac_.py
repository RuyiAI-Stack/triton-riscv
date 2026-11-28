import importlib

import pytest
import torch

from .igammac_ import igammac_


def _tolerance(dtype):
    # Tolerance source: same construction and bounds as the repo's
    # special_gammainc suite (series / continued fraction around the
    # x = a + 1 seam): 1e-4/1e-6 in fp32; the fp64 path must run the
    # branches in double precision (upstream igammac switches
    # COMPUTE_DTYPE), which lands at double ulp.
    if dtype == torch.float64:
        return 1e-9, 1e-12
    return 1e-4, 1e-6


@pytest.mark.parametrize("shape", [(512,), (1023,), (8, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_igammac_inplace_matches_functional_oracle(shape, dtype):
    torch.manual_seed(0)
    a_orig = torch.rand(shape, dtype=dtype) * 9.0 + 0.5
    x = torch.rand(shape, dtype=dtype) * 20.0
    a = a_orig.clone()

    ret = igammac_(a, x)

    assert ret is a
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(
        a, torch.special.gammaincc(a_orig, x), rtol=rtol, atol=atol
    )
    # self was mutated in place, not rebound to a fresh tensor
    assert not torch.equal(a, a_orig)


def test_igammac_inplace_boundaries_and_specials():
    # ATen contract probed on torch CPU fp32 (matches the functional
    # oracle torch.special.gammaincc): Q(a, 0) = 1, x = inf and large x
    # underflow to 0, x < 0 = nan, a < 0 = nan, nan propagates.
    a = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 0.5, 0.5, 1.0, float("nan")])
    x = torch.tensor(
        [
            0.0,
            -1.0,
            float("-inf"),
            float("nan"),
            float("inf"),
            1.0,
            1e10,
            float("inf"),
            1e10,
            1.0,
        ]
    )
    a_orig = a.clone()

    ret = igammac_(a, x)

    assert ret is a
    torch.testing.assert_close(
        a, torch.special.gammaincc(a_orig, x), rtol=0, atol=0, equal_nan=True
    )
    assert a[0].item() == 1.0
    assert a[4].item() == 0.0 and a[6].item() == 0.0


def test_igammac_inplace_non_contiguous():
    torch.manual_seed(0)
    base = torch.rand(8, 16) * 9.0 + 0.5
    view = base[::2, ::2]
    assert not view.is_contiguous()
    x = torch.rand(4, 8) * 20.0
    view_orig = view.clone()

    ret = igammac_(view, x)

    assert ret is view
    torch.testing.assert_close(
        view, torch.special.gammaincc(view_orig, x), rtol=1e-4, atol=1e-6
    )
    # the view's storage itself was rewritten
    assert not torch.equal(view, view_orig)


def test_igammac_inplace_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.igammac_")
    launches = []
    real_run = mod.igammac_func.run  # kernel name per upstream

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.igammac_func, "run", spy_run)
    a = torch.rand(1024) * 4.0 + 0.5
    a_orig = a.clone()
    x = torch.rand(1024) * 10.0
    ret = igammac_(a, x)
    assert len(launches) >= 1, "igammac_ must launch the Triton kernel"
    torch.testing.assert_close(
        ret, torch.special.gammaincc(a_orig, x), rtol=1e-4, atol=1e-6
    )
