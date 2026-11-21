import importlib

import torch

from .is_same_size import is_same_size


def test_is_same_size_true_cases():
    assert is_same_size(torch.empty(2, 3), torch.empty(2, 3)) is True
    assert is_same_size(torch.empty(4), torch.empty(4)) is True
    assert is_same_size(torch.empty(2, 3, 4), torch.empty(2, 3, 4)) is True


def test_is_same_size_false_cases():
    assert is_same_size(torch.empty(2, 3), torch.empty(3, 2)) is False
    assert is_same_size(torch.empty(4), torch.empty(5)) is False
    assert is_same_size(torch.empty(1, 4), torch.empty(4)) is False  # rank mismatch


def test_is_same_size_zero_dim():
    assert is_same_size(torch.tensor(1.0), torch.tensor(2.0)) is True
    assert is_same_size(torch.tensor(1.0), torch.empty(1)) is False


def test_is_same_size_matches_aten():
    for a, b in [
        ((2, 3), (2, 3)),
        ((2, 3), (3, 2)),
        ((5,), (5,)),
        ((2, 3, 4), (2, 3, 5)),
    ]:
        ta, tb = torch.empty(a), torch.empty(b)
        assert is_same_size(ta, tb) == torch.ops.aten.is_same_size(ta, tb)


def test_is_same_size_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.is_same_size")
    launches = []
    real_run = mod.is_same_size_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.is_same_size_kernel, "run", spy_run)
    result = is_same_size(torch.empty(2, 3, 4), torch.empty(2, 3, 4))
    assert len(launches) >= 1
    assert result is True
    assert is_same_size(torch.empty(2, 3, 4), torch.empty(2, 3, 5)) is False
