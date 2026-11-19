import importlib

import pytest
import torch

from .special_logit import special_logit, special_logit_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_special_logit(dtype, size):
    torch.manual_seed(0)
    x = torch.rand(size, dtype=dtype)

    ref = torch.special.logit(x)
    tri = special_logit(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: logit = log(x / (1 - x)) in fp32; the tails near 0
    # and 1 lose precision to the division and the log, so 1e-5 relative
    # covers fp32 while fp16 narrows once (1e-3).
    if dtype == torch.float16:
        torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_logit_tails():
    # Values hugging both ends of (0, 1): the branch has to stay finite
    # and keep relative accuracy until fp32 resolution runs out.
    x = torch.tensor(
        [1e-7, 1e-4, 0.01, 0.5, 0.99, 0.9999, 1 - 1e-7], dtype=torch.float32
    )

    ref = torch.special.logit(x)
    tri = special_logit(x)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)
    assert tri[3].item() == 0.0


def test_special_logit_special_values():
    # ATen contract probed on torch CPU fp32: logit(0) = -inf,
    # logit(1) = +inf, values outside [0, 1] = nan, nan stays nan.
    x = torch.tensor([0.0, 1.0, -0.1, 1.1, float("nan"), float("inf")])

    ref = torch.special.logit(x)
    tri = special_logit(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == float("-inf") and tri[1].item() == float("inf")


def test_special_logit_eps():
    # eps clamps the input before the transform (probed on torch CPU):
    # logit(0, eps=1e-3) == logit(1e-3), logit(1, eps=1e-3) == logit(1-1e-3).
    x = torch.tensor([0.0, 1.0, 0.5], dtype=torch.float32)

    ref = torch.special.logit(x, eps=1e-3)
    tri = special_logit(x, eps=1e-3)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_logit_out_variant():
    x = torch.rand(256, dtype=torch.float32) * 0.98 + 0.01
    out = torch.empty_like(x)

    res = special_logit_out(x, None, out)
    assert res is out
    torch.testing.assert_close(out, torch.special.logit(x), rtol=1e-5, atol=1e-5)


def test_special_logit_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_logit")
    launches = []
    real_run = mod.special_logit_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.special_logit_kernel, "run", spy_run)
    x = torch.rand(1024, dtype=torch.float32) * 0.9 + 0.05
    out = mod.special_logit(x)
    assert len(launches) >= 1, "special_logit must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.logit(x), rtol=1e-5, atol=1e-5)
