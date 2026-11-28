import importlib

import pytest
import torch

from .atleast_1d import _atleast_1d_materialize, atleast_1d
from .atleast_3d import _atleast_3d_materialize, atleast_3d


def test_atleast_1d_matches_aten_and_aliases():
    x0 = torch.tensor(3.5)
    out0 = atleast_1d(x0)
    assert out0.shape == (1,)
    assert out0.item() == 3.5

    x = torch.arange(2 * 3).reshape(2, 3)
    out = atleast_1d(x)
    assert out is x  # rank >= 1 aliases the input (reference contract)


def test_atleast_1d_sequence_form():
    outs = atleast_1d([torch.tensor(1.0), torch.arange(3)])
    assert outs[0].shape == (1,)
    assert outs[1].shape == (3,)
    assert atleast_1d([]) == []
    with pytest.raises(TypeError):
        atleast_1d(["nope"])


def test_atleast_1d_bad_type_raises():
    with pytest.raises(TypeError, match="expected a Tensor"):
        atleast_1d(3.5)


def test_atleast_3d_rank_promotion():
    x0 = torch.tensor(7.0)
    assert atleast_3d(x0).shape == (1, 1, 1)
    x1 = torch.arange(4)
    out1 = atleast_3d(x1)
    assert out1.shape == (1, 4, 1)
    x2 = torch.arange(6).reshape(2, 3)
    assert atleast_3d(x2).shape == (2, 3, 1)
    x3 = torch.arange(8).reshape(2, 2, 2)
    assert atleast_3d(x3) is x3  # alias for rank >= 3


def test_atleast_3d_matches_aten_values():
    for shape in [(), (4,), (2, 3), (2, 2, 2)]:
        x = torch.randn(shape) if shape else torch.tensor(1.5)
        torch.testing.assert_close(atleast_3d(x), torch.atleast_3d(x), rtol=0, atol=0)


def test_atleast_3d_sequence_form():
    outs = atleast_3d([torch.arange(3), torch.arange(4).reshape(2, 2)])
    assert outs[0].shape == (1, 3, 1)
    assert outs[1].shape == (2, 2, 1)
    assert atleast_3d(()) == []


def test_atleast_1d_materialize_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.atleast_1d")
    launches = []
    real_run = mod._atleast_1d_copy_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._atleast_1d_copy_kernel, "run", spy_run)
    x = torch.arange(2 * 3, dtype=torch.float64).reshape(2, 3)
    out = _atleast_1d_materialize(x)
    assert len(launches) >= 1
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    assert out.data_ptr() != x.data_ptr()


def test_atleast_3d_materialize_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.atleast_3d")
    launches = []
    real_run = mod._atleast_3d_copy_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._atleast_3d_copy_kernel, "run", spy_run)
    x = torch.arange(2 * 3, dtype=torch.float64).reshape(2, 3)
    out = _atleast_3d_materialize(x)
    assert len(launches) >= 1
    torch.testing.assert_close(out, x, rtol=0, atol=0)
