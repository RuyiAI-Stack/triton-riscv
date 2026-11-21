import importlib

import pytest
import torch

from .resize_as import resize_as, resize_as_


def test_resize_as_same_shape_returns_self():
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    template = torch.empty(2, 3)
    assert resize_as(x, template) is x


def test_resize_as_reshapes_with_copy():
    x = torch.arange(6, dtype=torch.float64)
    template = torch.empty(2, 3)
    out = resize_as(x, template)
    assert out.shape == (2, 3)
    assert out is not x
    torch.testing.assert_close(out, x.reshape(2, 3), rtol=0, atol=0)


def test_resize_as_numel_mismatch_raises():
    x = torch.arange(6)
    template = torch.empty(2, 2)
    with pytest.raises(RuntimeError, match="preserving the number of elements"):
        resize_as(x, template)
    with pytest.raises(RuntimeError, match="preserving the number of elements"):
        resize_as_(x, template)


def test_resize_as_inplace_set_semantics():
    x = torch.arange(6, dtype=torch.float64)
    template = torch.empty(3, 2)
    ret = resize_as_(x, template)
    assert ret is x
    assert x.shape == (3, 2)
    torch.testing.assert_close(
        x.reshape(-1), torch.arange(6, dtype=torch.float64), rtol=0, atol=0
    )


def test_resize_as_empty():
    x = torch.empty(0, dtype=torch.float64)
    template = torch.empty(0, 5)
    out = resize_as(x, template)
    assert out.shape == (0, 5)


def test_resize_as_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.resize_as")
    launches = []
    real_run = mod._resize_as_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._resize_as_kernel, "run", spy_run)
    x = torch.arange(6, dtype=torch.float64)
    out = resize_as(x, torch.empty(2, 3))
    assert len(launches) >= 1
    torch.testing.assert_close(out, x.reshape(2, 3), rtol=0, atol=0)
