import importlib

import torch

from .resize import resize, resize_


def test_resize_shrink_copies_prefix():
    x = torch.arange(6, dtype=torch.float64)
    out = resize(x, (4,))
    assert out.shape == (4,)
    torch.testing.assert_close(out, x[:4], rtol=0, atol=0)


def test_resize_same_numel_new_shape():
    x = torch.arange(6, dtype=torch.float64)
    out = resize(x, (2, 3))
    assert out.shape == (2, 3)
    torch.testing.assert_close(out, x.reshape(2, 3), rtol=0, atol=0)


def test_resize_empty_inputs():
    x = torch.empty(0, dtype=torch.float64)
    out = resize(x, (3,))
    assert out.shape == (3,)
    out2 = resize(torch.arange(4), (0, 2))
    assert out2.shape == (0, 2)


def test_resize_memory_format_accepted_but_ignored():
    x = torch.arange(6, dtype=torch.float64)
    out = resize(x, (6,), memory_format=torch.contiguous_format)
    torch.testing.assert_close(out, x, rtol=0, atol=0)


def test_resize_inplace_set_semantics():
    x = torch.arange(6, dtype=torch.float64)
    ret = resize_(x, (2, 3))
    assert ret is x
    assert x.shape == (2, 3)
    # set_ re-views the same storage: the values follow the flat layout.
    torch.testing.assert_close(
        x.reshape(-1), torch.arange(6, dtype=torch.float64), rtol=0, atol=0
    )


def test_resize_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.resize")
    launches = []
    real_run = mod._resize_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._resize_kernel, "run", spy_run)
    x = torch.arange(10, dtype=torch.float64)
    out = resize(x, (4,))
    assert len(launches) >= 1
    torch.testing.assert_close(out, x[:4], rtol=0, atol=0)
