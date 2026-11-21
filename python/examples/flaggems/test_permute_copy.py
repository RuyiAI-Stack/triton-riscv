import importlib

import torch

from .permute_copy import permute_copy


def test_permute_copy_matches_aten_2d():
    x = torch.arange(3 * 5, dtype=torch.float64).reshape(3, 5)
    for dims in [(1, 0), (0, 1)]:
        out = permute_copy(x, dims)
        ref = torch.ops.aten.permute_copy(x, dims)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=str(dims))


def test_permute_copy_matches_aten_3d_all_perms():
    x = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    import itertools

    for dims in itertools.permutations(range(3)):
        out = permute_copy(x, dims)
        ref = torch.ops.aten.permute_copy(x, dims)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=str(dims))


def test_permute_copy_negative_dims():
    x = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    out = permute_copy(x, (-1, 0, -2))
    ref = torch.ops.aten.permute_copy(x, (2, 0, 1))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_permute_copy_1d_identity():
    x = torch.arange(6, dtype=torch.float64)
    out = permute_copy(x, (0,))
    torch.testing.assert_close(out, x, rtol=0, atol=0)


def test_permute_copy_rank4_fallback():
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5)
    out = permute_copy(x, (3, 1, 2, 0))
    ref = torch.ops.aten.permute_copy(x, (3, 1, 2, 0))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_permute_copy_rank0_clone():
    x = torch.tensor(3.5)
    out = permute_copy(x, ())
    assert out.item() == 3.5
    assert out is not x


def test_permute_copy_strided_input():
    x = torch.randn(3, 4).t()  # (4, 3) non-contiguous
    out = permute_copy(x, (1, 0))
    ref = torch.ops.aten.permute_copy(x, (1, 0))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_permute_copy_empty():
    x = torch.empty(2, 0, 3)
    out = permute_copy(x, (2, 0, 1))
    assert out.shape == (3, 2, 0)


def test_permute_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.permute_copy")
    launches = []
    real_run = mod._permute_copy_kernel_3d.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._permute_copy_kernel_3d, "run", spy_run)
    x = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    out = permute_copy(x, (2, 0, 1))
    assert len(launches) >= 1
    torch.testing.assert_close(out, x.permute(2, 0, 1).contiguous(), rtol=0, atol=0)
    assert out.shape == (4, 2, 3)
