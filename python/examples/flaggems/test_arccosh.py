import pytest
import torch

from .arccosh import arccosh, arccosh_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_arccosh(size):
    torch.manual_seed(0)
    # arccosh domain is [1, inf); sample values in [1, 10).
    x = torch.rand(size, device="cpu", dtype=torch.float32) * 9.0 + 1.0

    ref_out = torch.arccosh(x)
    tri_out = arccosh(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_arccosh_out(size):
    torch.manual_seed(0)
    x = torch.rand(size, device="cpu", dtype=torch.float32) * 9.0 + 1.0
    out = torch.empty_like(x)

    ref_out = torch.arccosh(x)
    acosh_ret = arccosh_out(x, out)

    assert acosh_ret is out, "arccosh_out must return the caller's tensor"
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        # fp16: kernel computes in fp32 and narrows once (repo test_acosh
        # precedent).  fp64: libdevice.acosh runs the full double path.
        (torch.float16, 1e-3, 1e-3),
        (torch.float64, 1e-6, 1e-6),
    ],
)
def test_arccosh_other_float(dtype, rtol, atol):
    torch.manual_seed(0)
    x = (torch.rand(128, device="cpu", dtype=torch.float32) * 9.0 + 1.0).to(dtype)

    ref_out = torch.arccosh(x)
    tri_out = arccosh(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=rtol, atol=atol)


def test_arccosh_bf16():
    torch.manual_seed(0)
    x = (torch.rand(128, device="cpu", dtype=torch.float32) * 9.0 + 1.0).to(
        torch.bfloat16
    )

    ref_out = torch.arccosh(x)
    tri_out = arccosh(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)


def test_arccosh_domain_boundary():
    # In-domain edges: exactly 1.0 (-> 0.0), large values, +inf (-> +inf).
    x = torch.tensor(
        [1.0, 2.0, 1e30, 3.0e38, float("inf")], dtype=torch.float32
    )

    ref_out = torch.arccosh(x)
    tri_out = arccosh(x)

    assert tri_out[0].item() == 0.0, "arccosh(1.0) must be exactly 0.0"
    assert tri_out[4].item() == float("inf")
    torch.testing.assert_close(tri_out[:4], ref_out[:4], rtol=1e-4, atol=1e-4)


def test_arccosh_out_of_domain_nan():
    # Domain is [1, inf): everything below 1 (and NaN inputs) yields NaN.
    x = torch.tensor(
        [0.9999999, 0.5, 0.0, -1.0, -3.0, float("-inf"), float("nan")],
        dtype=torch.float32,
    )

    tri_out = arccosh(x)

    assert torch.isnan(tri_out).all()
    assert torch.isnan(torch.arccosh(x)).all()


def test_arccosh_out_dtype_mismatch():
    torch.manual_seed(0)
    x = torch.rand(128, device="cpu", dtype=torch.float32) * 9.0 + 1.0
    out = torch.empty(128, dtype=torch.float64)

    ref_out = torch.arccosh(x)
    ret = arccosh_out(x, out)

    assert ret is out, "arccosh_out must return the caller's tensor"
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, ref_out.to(torch.float64), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(0,), (3, 0)])
def test_arccosh_empty(shape):
    torch.manual_seed(0)
    x = torch.rand(shape, device="cpu", dtype=torch.float32) * 9.0 + 1.0

    tri_out = arccosh(x)

    assert tri_out.shape == x.shape
    assert tri_out.numel() == 0


def test_arccosh_non_contiguous():
    torch.manual_seed(0)
    x = (torch.rand(64, 32, device="cpu", dtype=torch.float32) * 9.0 + 1.0).t()
    assert not x.is_contiguous()

    torch.testing.assert_close(
        arccosh(x), torch.arccosh(x), rtol=1e-4, atol=1e-4
    )


def test_arccosh_integer_promotes():
    # Upstream INT_TO_FLOAT promotion: integer inputs come out as the
    # default float dtype (torch.arccosh itself rejects integer tensors).
    x = torch.arange(1, 9, dtype=torch.int32)

    tri_out = arccosh(x)

    assert tri_out.dtype == torch.get_default_dtype()
    torch.testing.assert_close(
        tri_out, torch.arccosh(x.to(torch.float32)), rtol=1e-4, atol=1e-4
    )


def test_arccosh_triton_path(monkeypatch):
    # Smoke: the fp32 main path must launch the Triton kernel.
    import importlib

    mod = importlib.import_module("flaggems.arccosh")

    launches = []
    real_run = mod.arccosh_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.arccosh_kernel, "run", spy_run)
    x = torch.rand(1024, device="cpu", dtype=torch.float32) * 9.0 + 1.0
    out = mod.arccosh(x)
    assert len(launches) == 1, "arccosh(A) must go through the Triton kernel"
    torch.testing.assert_close(out, torch.arccosh(x), rtol=1e-4, atol=1e-4)
