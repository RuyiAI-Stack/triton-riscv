import importlib
import math

import pytest
import torch

from ._fused_dropout import _fused_dropout


def test_fused_dropout_output_mask_consistency():
    torch.manual_seed(0)
    x = torch.randn(10000)
    p = 0.3

    out, mask = _fused_dropout(x, p)

    assert mask.dtype == torch.uint8
    assert bool(((mask == 0) | (mask == 1)).all())
    assert out.dtype == x.dtype and out.shape == x.shape
    # The output must be exactly the returned mask rescaled by 1/(1-p).
    expected = x * mask.to(x.dtype) / (1 - p)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-7)


def test_fused_dropout_p0_and_p1_edges():
    torch.manual_seed(0)
    x = torch.randn(64)

    out, mask = _fused_dropout(x, 0.0)
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    assert bool((mask == 1).all())

    out, mask = _fused_dropout(x, 1.0)
    assert bool((out == 0).all())
    assert bool((mask == 0).all())


def test_fused_dropout_strided_input_layout():
    # The kernel writes flat offsets: a non-contiguous input must still come
    # back with the values of its contiguous copy (result is contiguous).
    torch.manual_seed(0)
    x = torch.randn(16, 64).t()
    assert not x.is_contiguous()

    out, mask = _fused_dropout(x, 0.4)

    assert out.is_contiguous() and mask.is_contiguous()
    expected = x.contiguous() * mask.to(x.dtype) / 0.6
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-7)


def test_fused_dropout_empty():
    x = torch.empty(0, dtype=torch.float32)
    out, mask = _fused_dropout(x, 0.5)
    assert out.shape == (0,) and mask.shape == (0,) and mask.dtype == torch.uint8


def test_fused_dropout_rejects_nonfloat_and_bad_generator():
    with pytest.raises(RuntimeError):
        _fused_dropout(torch.arange(8), 0.5)
    with pytest.raises(TypeError):
        _fused_dropout(torch.randn(8), 0.5, generator=object())


def test_fused_dropout_rate_and_determinism():
    torch.manual_seed(0)
    n = 50000
    p = 0.4
    x = torch.randn(n)

    _, mask = _fused_dropout(x, p)
    kept = float(mask.float().mean())
    sigma = math.sqrt(p * (1 - p) / n)
    assert abs(kept - (1 - p)) < 4 * sigma + 1e-3, f"keep rate {kept}"

    torch.manual_seed(1234)
    out1, mask1 = _fused_dropout(x, p)
    torch.manual_seed(1234)
    out2, mask2 = _fused_dropout(x, p)
    assert torch.equal(mask1, mask2) and torch.equal(out1, out2)


def test_fused_dropout_fp64_row():
    torch.manual_seed(0)
    x = torch.randn(4096, dtype=torch.float64)
    out, mask = _fused_dropout(x, 0.25)
    assert out.dtype == torch.float64 and mask.dtype == torch.uint8
    expected = x * mask.to(x.dtype) / 0.75
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)


def test_fused_dropout_accepts_valid_generator():
    # A real Generator must be accepted (only foreign objects are rejected).
    gen = torch.Generator()
    gen.manual_seed(7)
    x = torch.randn(64)
    out, mask = _fused_dropout(x, 0.5, generator=gen)
    assert out.shape == x.shape and mask.dtype == torch.uint8


def test_fused_dropout_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._fused_dropout")
    launches = []
    real_run = mod._fused_dropout_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._fused_dropout_kernel, "run", spy_run)
    x = torch.randn(1024)
    out, mask = mod._fused_dropout(x, 0.5)
    assert len(launches) >= 1, "_fused_dropout must launch the Triton kernel"
    expected = x * mask.to(x.dtype) / 0.5
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-7)
