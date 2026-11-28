import importlib

import pytest
import torch

from .special_entr import special_entr


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_special_entr(dtype, size):
    torch.manual_seed(0)
    x = torch.rand(size, dtype=dtype) * 10.0

    ref = torch.special.entr(x)
    tri = special_entr(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: entr = -x log(x) on the positive domain; the kernel
    # computes the closed form in fp32 (1e-6 for fp32/fp64, 1e-3 after the
    # single fp16 narrow) and handles x <= 0 exactly, so no tolerance slack
    # is needed away from zero.
    if dtype == torch.float16:
        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-6)


def test_special_entr_negative_inputs():
    # x < 0 -> -inf, not nan (probed on torch CPU fp32), and every row is
    # an exact sentinel, so the comparison is bit-exact.
    x = torch.tensor(
        [-1e-6, -0.5, -1.0, -100.0, 0.0, -0.0, 1.0, float("inf"), float("nan")]
    )

    ref = torch.special.entr(x)
    tri = special_entr(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == float("-inf")
    assert tri[3].item() == float("-inf")
    assert tri[4].item() == 0.0 and tri[6].item() == 0.0
    assert tri[7].item() == float("-inf")


def test_special_entr_small_positive():
    # -x log(x) peaks at x = 1/e and goes to 0 as x -> 0; tiny inputs must
    # not flush to a wrong sign.
    x = torch.tensor([1e-30, 1e-10, 1e-5, 1e-2, 0.3678794, 1.0], dtype=torch.float32)

    ref = torch.special.entr(x)
    tri = special_entr(x)

    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-30)


def test_special_entr_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_entr")
    launches = []
    real_run = mod.special_entr_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.special_entr_func, "run", spy_run)
    x = torch.rand(1024, dtype=torch.float32) * 5.0
    out = mod.special_entr(x)
    assert len(launches) >= 1, "special_entr must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.entr(x), rtol=1e-6, atol=1e-6)
