import importlib

import pytest
import torch

from ._chunk_cat import chunk_cat


def test_chunk_cat_single_tensor_1d_matches_aten():
    x = torch.arange(10, dtype=torch.float64)
    out = chunk_cat([x], 0, 3)
    ref = torch.ops.aten._chunk_cat([x], 0, 3)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == (3, 4)  # chunks zero-padded to ceil(10/3)=4


def test_chunk_cat_single_tensor_2d():
    x = torch.arange(3 * 7, dtype=torch.float64).reshape(3, 7)
    for dim in (0, 1):
        out = chunk_cat([x], dim, 2)
        ref = torch.ops.aten._chunk_cat([x], dim, 2)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"dim={dim}")


def test_chunk_cat_single_tensor_3d():
    x = torch.arange(2 * 5 * 3, dtype=torch.float64).reshape(2, 5, 3)
    out = chunk_cat([x], 1, 2)
    ref = torch.ops.aten._chunk_cat([x], 1, 2)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_chunk_cat_multi_tensor_matches_aten():
    a = torch.arange(6, dtype=torch.float64)
    b = torch.arange(6, dtype=torch.float64) * 10
    for num_chunks in (2, 3, 4):
        out = chunk_cat([a, b], 0, num_chunks)
        ref = torch.ops.aten._chunk_cat([a, b], 0, num_chunks)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"n={num_chunks}")
    a2 = torch.arange(2 * 3, dtype=torch.float64).reshape(2, 3)
    b2 = a2 * 10
    for dim in (0, 1):
        out = chunk_cat([a2, b2], dim, 2)
        ref = torch.ops.aten._chunk_cat([a2, b2], dim, 2)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"dim={dim}")


def test_chunk_cat_empty_list_raises():
    with pytest.raises(ValueError, match="non-empty"):
        chunk_cat([], 0, 2)


def test_chunk_cat_bad_num_chunks_raises():
    x = torch.arange(6)
    with pytest.raises(ValueError, match="must be positive"):
        chunk_cat([x], 0, 0)


def test_chunk_cat_dim_out_of_range():
    x = torch.arange(6)
    with pytest.raises(IndexError, match="out of range"):
        chunk_cat([x], 1, 2)


def test_chunk_cat_zero_dim_input_raises():
    x = torch.tensor(1.0)
    with pytest.raises(ValueError, match="at least 1 dimension"):
        chunk_cat([x], 0, 2)


def test_chunk_cat_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._chunk_cat")
    launches = []
    real_run = mod._chunk_cat_kernel_1d.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._chunk_cat_kernel_1d, "run", spy_run)
    x = torch.arange(10, dtype=torch.float64)
    out = chunk_cat([x], 0, 3)
    assert len(launches) >= 1
    torch.testing.assert_close(out[0], x[0:4], rtol=0, atol=0)
    torch.testing.assert_close(
        out[2, 2:], torch.zeros(2, dtype=torch.float64), rtol=0, atol=0
    )
