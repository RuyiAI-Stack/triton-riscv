import importlib

import pytest
import torch

from .igammac import igammac, igammac_out


def _tolerance(dtype):
    # Tolerance source: same construction and bounds as the repo's
    # special_gammainc suite (series / continued fraction around the
    # x = a + 1 seam): 1e-4/1e-6 in fp32; upstream's igammac kernel
    # already runs COMPUTE_DTYPE = float64 for fp64 outputs, which lands
    # at double ulp, so 1e-9/1e-12 keeps a wide margin.
    if dtype == torch.float64:
        return 1e-9, 1e-12
    return 1e-4, 1e-6


@pytest.mark.parametrize("shape", [(512,), (1023,), (8, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_igammac_matches_aten(shape, dtype):
    torch.manual_seed(0)
    a = torch.rand(shape, dtype=dtype) * 9.0 + 0.5
    x = torch.rand(shape, dtype=dtype) * 20.0

    ref = torch.special.gammaincc(a, x)
    tri = igammac(a, x)

    assert tri.dtype == ref.dtype
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


def test_igammac_branch_matrix():
    # x sweeps across the x = a + 1 branch switch and far past it,
    # including the large-a / large-x regime where Q underflows.
    a = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0, 80.0]).repeat_interleave(41)
    x = torch.arange(0.0, 20.5, 0.5).repeat(6)

    ref = torch.special.gammaincc(a, x)
    tri = igammac(a, x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-6)


def test_igammac_boundaries_and_specials():
    # ATen contract probed on torch CPU fp32: Q(a, 0) = 1, x = inf and
    # large x underflow to 0, x < 0 = nan (including -inf), a < 0 = nan,
    # nan propagates.
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

    ref = torch.special.gammaincc(a, x)
    tri = igammac(a, x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == 1.0
    assert tri[4].item() == 0.0 and tri[6].item() == 0.0 and tri[7].item() == 0.0
    assert bool(torch.isnan(tri[5]))


def test_igammac_non_contiguous():
    torch.manual_seed(0)
    a = torch.rand(4, 8) * 9.0 + 0.5
    x = torch.rand(4, 8) * 20.0
    a_v, x_v = a.t(), x.t()
    assert not a_v.is_contiguous() and not x_v.is_contiguous()

    ref = torch.special.gammaincc(a_v, x_v)
    tri = igammac(a_v, x_v)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-6)


def test_igammac_out_variant():
    torch.manual_seed(0)
    a = torch.rand(256, dtype=torch.float32) * 4.0 + 0.5
    x = torch.rand(256, dtype=torch.float32) * 10.0
    out = torch.empty_like(a)

    res = igammac(a, x, out=out)
    assert res is out
    torch.testing.assert_close(out, torch.special.gammaincc(a, x), rtol=1e-4, atol=1e-6)

    out2 = torch.empty_like(a)
    res2 = igammac_out(a, x, out2)
    assert res2 is out2
    torch.testing.assert_close(
        out2, torch.special.gammaincc(a, x), rtol=1e-4, atol=1e-6
    )


def test_igammac_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.igammac")
    launches = []
    real_run = mod.igammac_kernel.run  # kernel name per upstream

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.igammac_kernel, "run", spy_run)
    a = torch.rand(1024) * 4.0 + 0.5
    x = torch.rand(1024) * 10.0
    out = igammac(a, x)
    assert len(launches) >= 1, "igammac must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.gammaincc(a, x), rtol=1e-4, atol=1e-6)
