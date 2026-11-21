import importlib

import pytest
import torch

from .new_zeros import new_zeros, new_zeros_out


def test_new_zeros_inherits_and_zeros():
    self_t = torch.empty(3, dtype=torch.float64)
    out = new_zeros(self_t, (2, 3))
    assert out.dtype == torch.float64
    assert bool((out == 0).all())


def test_new_zeros_complex_and_bool():
    out_c = new_zeros(torch.empty(2, dtype=torch.complex128), (3, 4))
    assert out_c.dtype == torch.complex128
    assert bool((out_c == 0).all())
    out_b = new_zeros(torch.empty(2, dtype=torch.bool), (5,))
    assert not bool(out_b.any())


def test_new_zeros_strided_out_zeroed():
    # A non-contiguous out buffer zeros only its logical elements.
    buf = torch.ones(4, 6, dtype=torch.float64).t()  # (6, 4) strided view
    new_zeros_out(torch.empty(2), (6, 4), out=buf)
    assert bool((buf == 0).all())


def test_new_zeros_size_normalization():
    out = new_zeros(torch.empty(2), torch.tensor([2, 3]))
    assert out.shape == (2, 3)
    out2 = new_zeros(torch.empty(2), torch.Size([2, 3]))
    assert out2.shape == (2, 3)
    with pytest.raises(TypeError, match="sequence of integers"):
        new_zeros(torch.empty(2), 5)


def test_new_zeros_out_variant_resizes():
    out = torch.empty(6, dtype=torch.float32)
    ret = new_zeros_out(torch.empty(2), (2, 3), out=out)
    assert ret is out
    assert out.shape == (2, 3)
    assert bool((out == 0).all())


def test_new_zeros_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.new_zeros")
    launches = []
    real_run = mod._zero_bytes.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._zero_bytes, "run", spy_run)
    out = new_zeros(torch.empty(2, dtype=torch.float64), (3, 4))
    assert len(launches) >= 1
    assert bool((out == 0).all())
