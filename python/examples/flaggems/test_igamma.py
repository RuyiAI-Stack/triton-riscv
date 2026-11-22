import importlib

import pytest
import torch

from .igamma import igamma


def _tolerance(dtype):
    # Tolerance source: same construction as the repo's special_gammainc
    # port (power series below x = a + 1, Lentz continued fraction above);
    # its suite measured 1e-4/1e-6 across the branch seam in fp32.  fp64
    # must run the same branches in double precision (upstream igammac
    # already switches COMPUTE_DTYPE), which lands at double ulp, so
    # 1e-9/1e-12 keeps a wide margin.
    if dtype == torch.float64:
        return 1e-9, 1e-12
    return 1e-4, 1e-6


@pytest.mark.parametrize("shape", [(512,), (1023,), (8, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_igamma_matches_aten(shape, dtype):
    torch.manual_seed(0)
    a = torch.rand(shape, dtype=dtype) * 9.0 + 0.5
    x = torch.rand(shape, dtype=dtype) * 20.0

    ref = torch.special.gammainc(a, x)
    tri = igamma(a, x)

    assert tri.dtype == ref.dtype
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


def test_igamma_branch_matrix():
    # a fixed, x varying and both varying: the branch switch at x = a + 1
    # must be seamless (x sweeps far past a + 1, including large x).
    a = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0]).repeat_interleave(41)
    x = torch.arange(0.0, 20.5, 0.5).repeat(5)

    ref = torch.special.gammainc(a, x)
    tri = igamma(a, x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-6)


def test_igamma_boundaries_and_specials():
    # ATen contract probed on torch CPU fp32: P(a, 0) = 0, x < 0 = nan
    # (including -inf), a < 0 = nan, x = inf and large x saturate at 1,
    # nan propagates.
    a = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, 0.5, 0.5, 1.0, float("nan")])
    x = torch.tensor(
        [0.0, -1.0, float("-inf"), float("nan"), 1.0, float("inf"), 1e10, 1e10, 1.0]
    )

    ref = torch.special.gammainc(a, x)
    tri = igamma(a, x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == 0.0
    assert tri[5].item() == 1.0 and tri[6].item() == 1.0 and tri[7].item() == 1.0
    assert bool(torch.isnan(tri[4]))


def test_igamma_non_contiguous():
    torch.manual_seed(0)
    a = torch.rand(4, 8) * 9.0 + 0.5
    x = torch.rand(4, 8) * 20.0
    a_v, x_v = a.t(), x.t()
    assert not a_v.is_contiguous() and not x_v.is_contiguous()

    ref = torch.special.gammainc(a_v, x_v)
    tri = igamma(a_v, x_v)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-6)


def test_igamma_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.igamma")
    launches = []
    real_run = mod.igamma_kernel.run  # kernel name per upstream

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.igamma_kernel, "run", spy_run)
    a = torch.rand(1024) * 4.0 + 0.5
    x = torch.rand(1024) * 10.0
    out = igamma(a, x)
    assert len(launches) >= 1, "igamma must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.gammainc(a, x), rtol=1e-4, atol=1e-6)
