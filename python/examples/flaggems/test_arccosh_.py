import pytest
import torch

from .arccosh_ import arccosh_


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_arccosh_inplace(size):
    torch.manual_seed(0)
    # arccosh domain is [1, inf); sample values in [1, 10).
    x = torch.rand(size, device="cpu", dtype=torch.float32) * 9.0 + 1.0

    ref = x.clone()
    torch.arccosh_(ref)

    tri = x.clone()
    out = arccosh_(tri)

    assert out is tri, "arccosh_ should return the same tensor"
    torch.testing.assert_close(tri, ref.t(), rtol=1e-4, atol=1e-4)


def test_arccosh_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.rand(32, 64, device="cpu", dtype=torch.float32) * 9.0 + 1.0
    x_sliced = x[::2, ::2]  # non-contiguous

    ref = x_sliced.clone()
    torch.arccosh_(ref)

    # .t() keeps the view genuinely non-contiguous (a .clone() of a
    # strided view re-packs contiguous, which would only exercise
    # the fast path).
    tri = x_sliced.t()
    assert not tri.is_contiguous()
    out = arccosh_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref.t(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_arccosh_inplace_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.rand(128, device="cpu", dtype=torch.float32) * 9.0 + 1.0).to(dtype)

    ref = x.clone()
    torch.arccosh_(ref)

    tri = x.clone()
    out = arccosh_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref.t(), rtol=1e-3, atol=1e-3)


def test_arccosh__bf16():
    torch.manual_seed(0)
    x = (torch.rand(128) * 9.0 + 1.0).to(torch.bfloat16)

    ref = torch.arccosh(x)

    tri = x.clone()
    out = arccosh_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref.t(), rtol=1e-2, atol=1e-2)


def test_arccosh_inplace_boundary():
    # In-domain edges: exactly 1.0 (-> 0.0), plain values, +inf (-> +inf).
    x = torch.tensor([1.0, 2.0, 1e30, 3.0e38, float("inf")], dtype=torch.float32)

    ref = x.clone()
    torch.arccosh_(ref)

    tri = x.clone()
    out = arccosh_(tri)

    assert out is tri
    assert tri[0].item() == 0.0, "arccosh(1.0) must be exactly 0.0"
    assert tri[4].item() == float("inf")
    torch.testing.assert_close(tri[:4], ref[:4], rtol=1e-4, atol=1e-4)


def test_arccosh_inplace_out_of_domain_nan():
    # Domain is [1, inf): everything below 1 (and NaN inputs) yields NaN.
    x = torch.tensor(
        [0.9999999, 0.5, 0.0, -1.0, -3.0, float("-inf"), float("nan")],
        dtype=torch.float32,
    )

    tri = x.clone()
    out = arccosh_(tri)

    assert out is tri
    assert torch.isnan(tri).all()
    assert torch.isnan(torch.arccosh(x)).all()


@pytest.mark.parametrize("shape", [(0,), (3, 0)])
def test_arccosh_inplace_empty(shape):
    x = torch.rand(shape, device="cpu", dtype=torch.float32) * 9.0 + 1.0

    out = arccosh_(x)

    assert out is x, "arccosh_ on an empty tensor must still return self"
    assert out.numel() == 0


def test_arccosh_triton_path(monkeypatch):
    # Smoke: the fp32 contiguous path must launch the Triton kernel.
    import importlib

    mod = importlib.import_module("flaggems.arccosh_")

    launches = []
    real_run = mod.arccosh_kernel_.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.arccosh_kernel_, "run", spy_run)
    x = torch.rand(1024, device="cpu", dtype=torch.float32) * 9.0 + 1.0
    ref = torch.arccosh(x)  # before the in-place call consumes x
    out = mod.arccosh_(x)
    assert len(launches) == 1, "arccosh_(A) must go through the Triton kernel"
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
