import importlib

import pytest
import torch

from .special_zeta import (
    special_zeta,
    special_zeta_out,
    special_zeta_scalar_tensor,
    special_zeta_scalar_tensor_out,
    special_zeta_tensor_scalar,
    special_zeta_tensor_scalar_out,
)


@pytest.mark.parametrize("size", [512, 1023])
def test_special_zeta(size):
    torch.manual_seed(0)
    x = torch.rand(size, dtype=torch.float32) * 9.0 + 1.0  # s in (1, 10)
    q = torch.rand(size, dtype=torch.float32) * 9.0 + 0.5

    ref = torch.special.zeta(x, q)
    tri = special_zeta(x, q)

    assert tri.dtype == ref.dtype
    # Tolerance source: Hurwitz zeta for s > 1 uses the Euler-Maclaurin
    # tail in fp32; 1e-5 relative covers it while staying tight enough to
    # catch a missing Bernoulli term.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_zeta_known_values():
    # zeta(2, q) = trigamma(q) and zeta(s, 1) = Riemann zeta(s): both are
    # pinned against torch rows the implementation must reproduce.
    q = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
    s = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)

    ref = torch.special.zeta(s, q)
    tri = special_zeta(s, q)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(tri[1], torch.tensor(1.6449341), rtol=1e-5, atol=1e-6)


def test_special_zeta_scalar_overloads():
    x = torch.rand(256, dtype=torch.float32) * 3.0 + 1.5

    ref_ts = torch.special.zeta(x, 2.0)
    tri_ts = special_zeta_tensor_scalar(x, 2.0)
    torch.testing.assert_close(tri_ts, ref_ts, rtol=1e-5, atol=1e-6)

    ref_st = torch.special.zeta(2.0, x)
    tri_st = special_zeta_scalar_tensor(2.0, x)
    torch.testing.assert_close(tri_st, ref_st, rtol=1e-5, atol=1e-6)


def test_special_zeta_out_overloads():
    x = torch.rand(256, dtype=torch.float32) * 3.0 + 1.5

    out = torch.empty_like(x)
    res = special_zeta_tensor_scalar_out(x, 2.0, out)
    assert res is not None
    torch.testing.assert_close(out, torch.special.zeta(x, 2.0), rtol=1e-5, atol=1e-6)

    out2 = torch.empty_like(x)
    res2 = special_zeta_scalar_tensor_out(2.0, x, out2)
    assert res2 is not None
    torch.testing.assert_close(out2, torch.special.zeta(2.0, x), rtol=1e-5, atol=1e-6)


def test_special_zeta_special_values():
    # ATen contract probed on torch CPU fp32: zeta(1, q) = +inf,
    # zeta(0, 1) = nan, zeta(-1, 1) = nan, zeta(0.5, 1) = nan,
    # zeta(inf, 1) = 1, nan stays nan.
    x = torch.tensor(
        [1.0, 0.0, -1.0, 0.5, float("inf"), float("nan")], dtype=torch.float32
    )
    q = torch.ones_like(x)

    ref = torch.special.zeta(x, q)
    tri = special_zeta(x, q)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == float("inf") and tri[4].item() == 1.0


def test_special_zeta_out_variant():
    x = torch.rand(256, dtype=torch.float32) * 3.0 + 1.5
    q = torch.rand(256, dtype=torch.float32) * 3.0 + 0.5
    out = torch.empty_like(x)

    res = special_zeta_out(x, q, out)
    assert res is out
    torch.testing.assert_close(out, torch.special.zeta(x, q), rtol=1e-5, atol=1e-6)


def test_special_zeta_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_zeta")
    launches = []
    real_run = mod.special_zeta_flat_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.special_zeta_flat_kernel, "run", spy_run)
    x = torch.rand(1024, dtype=torch.float32) * 3.0 + 1.5
    q = torch.rand(1024, dtype=torch.float32) * 3.0 + 0.5
    out = mod.special_zeta(x, q)
    assert len(launches) >= 1, "special_zeta must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.zeta(x, q), rtol=1e-5, atol=1e-6)
