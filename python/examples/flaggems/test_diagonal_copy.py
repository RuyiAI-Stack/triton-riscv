import importlib

import pytest
import torch

from .diagonal_copy import diagonal_copy


def test_diagonal_copy_matches_aten_default():
    x = torch.arange(3 * 4 * 5, dtype=torch.float64).reshape(3, 4, 5)
    out = diagonal_copy(x)
    ref = torch.ops.aten.diagonal_copy(x)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == (5, 3)  # remaining dims (5,) + min(3, 4) diagonal


def test_diagonal_copy_offset_and_dims():
    x = torch.arange(3 * 4 * 5, dtype=torch.float64).reshape(3, 4, 5)
    for offset, dim1, dim2 in [(1, 0, 2), (-1, 0, 2), (0, 1, 2), (2, 2, 0), (-2, 1, 0)]:
        out = diagonal_copy(x, offset=offset, dim1=dim1, dim2=dim2)
        ref = torch.ops.aten.diagonal_copy(x, offset, dim1, dim2)
        torch.testing.assert_close(
            out, ref, rtol=0, atol=0, msg=f"{offset},{dim1},{dim2}"
        )


def test_diagonal_copy_negative_dims_resolve():
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    out = diagonal_copy(x, offset=1, dim1=-2, dim2=-1)
    ref = torch.ops.aten.diagonal_copy(x, offset=1, dim1=1, dim2=2)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_diagonal_copy_is_value_not_view():
    x = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4)
    out = diagonal_copy(x)
    out[0] = -1.0
    assert x[0, 0] == 0.0  # writing the copy must not touch the source


def test_diagonal_copy_same_dim_raises():
    x = torch.randn(3, 3)
    with pytest.raises(ValueError, match="dim1 and dim2 must be different"):
        diagonal_copy(x, 0, 1, 1)


def test_diagonal_copy_out_of_range_offset_empty():
    # |offset| beyond the diagonal extent yields an empty diagonal.
    x = torch.arange(3 * 4, dtype=torch.float64).reshape(3, 4)
    out = diagonal_copy(x, offset=10)
    assert out.shape == (0,)
    out_neg = diagonal_copy(x, offset=-10)
    assert out_neg.shape == (0,)


def test_diagonal_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.diagonal_copy")
    launches = []
    real_run = mod._diagonal_copy_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._diagonal_copy_kernel, "run", spy_run)
    x = torch.arange(5 * 6, dtype=torch.float64).reshape(5, 6)
    out = diagonal_copy(x, offset=1)
    assert len(launches) >= 1, "diagonal_copy must launch the Triton kernel"
    ref = torch.diagonal(x, offset=1)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == ref.shape
