import importlib

import pytest
import torch

from .polygamma import polygamma, polygamma_, polygamma_out


def _tolerance(dtype):
    # Tolerance source: n = 0 (digamma) is the repo's special_psi port
    # (1e-5/1e-6 measured); n >= 1 adds the trigamma reflection or the
    # Cephes Hurwitz zeta scaled by n!, whose fp32 error grows with the
    # n! factor, so fp32 sits at 1e-4/1e-6.  The fp64 path must evaluate
    # in double precision (aten does), landing at double ulp: 1e-9/1e-12.
    if dtype == torch.float64:
        return 1e-9, 1e-12
    return 1e-4, 1e-6


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(512,), (1023,), (8, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_polygamma_matches_aten(n, shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype) * 5.0 + 1.0

    ref = torch.special.polygamma(n, x)
    tri = polygamma(n, x)

    assert tri.dtype == ref.dtype
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_polygamma_anchor_values(n):
    # x = 1 and x = 2.5 sweep every kernel branch upstream routes to:
    # n = 0 digamma, n = 1 trigamma, n >= 2 the Hurwitz zeta path.
    for x in (1.0, 2.5):
        tri = polygamma(n, torch.tensor([x]))
        ref = torch.special.polygamma(n, torch.tensor([x]))
        torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-6)


def test_polygamma_zero_pole_alternates_like_aten():
    # ATen contract probed on torch CPU fp32: polygamma(n, 0) is the
    # right-limit -- -inf for even n, +inf for odd n.
    x = torch.zeros(4)

    for n in range(4):
        tri = polygamma(n, x.clone())
        ref = torch.special.polygamma(n, x)
        torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
        expected = float("-inf") if n % 2 == 0 else float("inf")
        assert tri[0].item() == expected


def test_polygamma_negative_fractional_matches_aten():
    # Reflection covers x < 0 away from the poles; negative INTEGER x is
    # deliberately not pinned: aten's own fp32 values there are pole
    # noise (polygamma(1, -1) = 1.29e15 finite), not a stable contract.
    xs = torch.tensor([-0.5, -2.5])

    for n in (0, 1, 2):
        tri = polygamma(n, xs.clone())
        ref = torch.special.polygamma(n, xs)
        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-5)


def test_polygamma_special_values():
    # ATen contract probed on torch CPU fp32: trigamma decays to 0 at
    # +inf, nan propagates.
    xs = torch.tensor([float("inf"), float("nan")])

    tri = polygamma(1, xs.clone())
    ref = torch.special.polygamma(1, xs)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == 0.0


@pytest.mark.parametrize("n", [-1, -5])
def test_polygamma_negative_n_rejected_like_aten(n):
    # Exact aten error, probed on torch CPU 2.15 (identical to the
    # upstream wrapper's message): RuntimeError with
    # "polygamma(n, x) does not support negative n."
    x = torch.rand(8) * 4.0 + 1.0
    with pytest.raises(
        RuntimeError, match=r"polygamma\(n, x\) does not support negative n\."
    ):
        polygamma(n, x)
    with pytest.raises(
        RuntimeError, match=r"polygamma\(n, x\) does not support negative n\."
    ):
        polygamma_(x, n)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_polygamma_inplace_matches_functional_oracle(n):
    # NOTE the argument order: upstream's in-place entry is
    # polygamma_(A, n), mirroring the tensor method A.polygamma_(n).
    torch.manual_seed(0)
    x_orig = torch.rand(512) * 5.0 + 1.0
    x = x_orig.clone()

    ret = polygamma_(x, n)

    assert ret is x
    torch.testing.assert_close(
        x, torch.special.polygamma(n, x_orig), rtol=1e-4, atol=1e-6
    )
    # self was mutated in place, not rebound to a fresh tensor
    assert not torch.equal(x, x_orig)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_polygamma_out_variant(n):
    torch.manual_seed(0)
    x = torch.rand(512) * 5.0 + 1.0
    out = torch.empty_like(x)

    res = polygamma_out(n, x, out)

    assert res is out
    torch.testing.assert_close(out, torch.special.polygamma(n, x), rtol=1e-4, atol=1e-6)


def test_polygamma_non_contiguous():
    torch.manual_seed(0)
    base = torch.rand(8, 16) * 5.0 + 1.0
    view = base.t()
    assert not view.is_contiguous()

    ref = torch.special.polygamma(2, view)
    tri = polygamma(2, view)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-6)

    sliced = base[::2, ::2]
    assert not sliced.is_contiguous()
    sliced_orig = sliced.clone()
    ret = polygamma_(sliced, 2)
    assert ret is sliced
    torch.testing.assert_close(
        sliced, torch.special.polygamma(2, sliced_orig), rtol=1e-4, atol=1e-6
    )
    # the view's storage itself was rewritten
    assert not torch.equal(sliced, sliced_orig)


def test_polygamma_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.polygamma")
    launches = []
    # kernel name per upstream: n >= 2 routes to polygamma_zeta_func
    # (n = 0 / 1 go to the digamma / trigamma kernels); n = 4 pins the
    # zeta launch.
    real_run = mod.polygamma_zeta_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.polygamma_zeta_func, "run", spy_run)
    x = torch.rand(1024) * 5.0 + 1.0
    out = polygamma(4, x)
    assert len(launches) >= 1, "polygamma must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.polygamma(4, x), rtol=1e-4, atol=1e-6)
