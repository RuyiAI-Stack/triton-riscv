import importlib

import torch

from ._resize_output import _resize_output, _resize_output_


def test_resize_output_shrink_and_grow():
    x = torch.arange(6, dtype=torch.float64)
    small = _resize_output(x, [4], torch.device("cpu"))
    torch.testing.assert_close(small, x[:4], rtol=0, atol=0)
    big = _resize_output(x, [10], torch.device("cpu"))
    torch.testing.assert_close(big[:6], x, rtol=0, atol=0)


def test_resize_output_same_shape_same_device_is_reshape():
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    out = _resize_output(x, [3, 2], torch.device("cpu"))
    assert out.shape == (3, 2)
    torch.testing.assert_close(out.reshape(-1), x.reshape(-1), rtol=0, atol=0)


def test_resize_output_inplace_same_device():
    x = torch.arange(6, dtype=torch.float64)
    ret = _resize_output_(x, [3], torch.device("cpu"))
    assert ret is x
    assert x.shape == (3,)


def test_resize_output_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._resize_output")
    launches = []
    real_run = mod._resize_copy_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._resize_copy_kernel, "run", spy_run)
    x = torch.arange(6, dtype=torch.float64)
    out = _resize_output(x, [10], torch.device("cpu"))
    assert len(launches) >= 1
    torch.testing.assert_close(out[:6], x, rtol=0, atol=0)
    assert out.shape == (10,)
