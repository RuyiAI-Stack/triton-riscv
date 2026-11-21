import importlib

import pytest
import torch

from .special_log_ndtr import special_log_ndtr, special_log_ndtr_


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_special_log_ndtr(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * 5.0

    ref = torch.special.log_ndtr(x)
    tri = special_log_ndtr(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: log(ndtr(x)) computed in fp32; 1e-5 relative holds
    # away from the log's zero crossing, and the atol covers values near 0
    # where log(ndtr) is tiny.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_log_ndtr_deep_tail():
    # The far negative tail is the whole point of the log form: ndtr
    # underflows to 0 at x ~ -40, log_ndtr keeps ~-800 and must stay finite
    # and relative-accurate down to the fp32 floor.
    x = torch.tensor([-100.0, -80.0, -60.0, -40.0, -30.0, -20.0, -10.0, -5.0, 0.0, 3.0])

    ref = torch.special.log_ndtr(x)
    tri = special_log_ndtr(x)

    assert bool(torch.isfinite(tri[:6]).all()), (
        "log_ndtr must stay finite deep in the tail"
    )
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-5)


def test_special_log_ndtr_special_values():
    # ATen contract probed on torch CPU fp32: log_ndtr(-inf) = -inf,
    # log_ndtr(0) = -ln 2, log_ndtr(+inf) = -0.0, nan stays nan.
    x = torch.tensor(
        [float("-inf"), 0.0, float("inf"), float("nan")], dtype=torch.float32
    )

    ref = torch.special.log_ndtr(x)
    tri = special_log_ndtr(x)

    # log_ndtr(0) = -ln 2 is computed on both sides (torch through log1p,
    # the port through its own log1p), so the finite row compares at ulp
    # distance; the +/-inf and nan rows are exact in the explicit asserts.
    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-7, equal_nan=True)
    assert tri[0].item() == float("-inf")
    assert tri[2].item() == 0.0 and tri[3] != tri[3]
    assert abs(tri[1].item() + 0.6931472) < 1e-6


def test_special_log_ndtr_inplace():
    x = torch.randn(256, dtype=torch.float32) * 3.0
    ref = torch.special.log_ndtr(x)

    tri = x.clone()
    out = special_log_ndtr_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_log_ndtr_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_log_ndtr")
    launches = []
    real_run = mod.special_log_ndtr_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.special_log_ndtr_func, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 4.0
    out = mod.special_log_ndtr(x)
    assert len(launches) >= 1, "special_log_ndtr must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.log_ndtr(x), rtol=1e-5, atol=1e-6)
