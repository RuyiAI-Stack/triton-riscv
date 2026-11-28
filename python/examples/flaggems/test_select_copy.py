import importlib

import pytest
import torch

from .select_copy import select_copy


def test_select_copy_matches_aten():
    x = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    for dim, index in [(0, 1), (1, 0), (1, 2), (2, 3), (-1, 1), (1, -1)]:
        out = select_copy(x, dim, index)
        ref = torch.ops.aten.select_copy(x, dim, index)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"{dim},{index}")
        assert out.shape == ref.shape


def test_select_copy_strided_input():
    x = torch.randn(4, 6).t()  # (6, 4) non-contiguous
    assert not x.is_contiguous()
    out = select_copy(x, 1, 2)
    ref = torch.ops.aten.select_copy(x, 1, 2)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_select_copy_higher_rank_strided():
    x = torch.randn(2, 3, 4).permute(1, 2, 0)  # (3, 4, 2) non-contiguous
    out = select_copy(x, 1, 1)
    ref = torch.ops.aten.select_copy(x, 1, 1)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_select_copy_zero_dim_raises():
    x = torch.tensor(3.14)
    with pytest.raises(RuntimeError, match="cannot be applied to a 0-dim tensor"):
        select_copy(x, 0, 0)


def test_select_copy_index_out_of_range():
    x = torch.randn(3, 5)
    with pytest.raises(RuntimeError, match="index 5 out of range"):
        select_copy(x, 1, 5)
    with pytest.raises(RuntimeError, match="index -6 out of range"):
        select_copy(x, 1, -6)


def test_select_copy_empty_dim_raises_like_aten():
    # aten rejects any index on a size-0 dimension; the port mirrors that.
    x = torch.empty(3, 0, 5)
    with pytest.raises(RuntimeError, match="index 0 out of range"):
        select_copy(x, 1, 0)


def test_select_copy_is_value_copy():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = select_copy(x, 0, 1)
    out[0] = -1.0
    assert x[1, 0] == 3.0


def test_select_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.select_copy")
    launches = []
    real_run = mod._select_copy_strided_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._select_copy_strided_kernel, "run", spy_run)
    x = torch.randn(2, 3, 4).permute(1, 0, 2)
    out = select_copy(x, 1, 1)
    assert len(launches) >= 1, "select_copy must launch the strided Triton kernel"
    torch.testing.assert_close(out, x.select(1, 1), rtol=0, atol=0)
    assert out.shape == (3, 4)


def test_select_copy_contig_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.select_copy")
    launches = []
    real_run = mod._select_copy_contig_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._select_copy_contig_kernel, "run", spy_run)
    x = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    out = select_copy(x, 0, 1)
    assert len(launches) >= 1
    torch.testing.assert_close(out, x[1], rtol=0, atol=0)
