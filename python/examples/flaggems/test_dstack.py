import importlib

import pytest
import torch

from .dstack import dstack


def _spy_on(kernel, launches):
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    return spy_run


def test_dstack_1d_matches_aten():
    a = torch.arange(3, dtype=torch.float64)
    b = torch.arange(3, dtype=torch.float64) * 10
    out = dstack([a, b])
    ref = torch.dstack([a, b])
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == ref.shape == (1, 3, 2)


def test_dstack_0d_and_2d_match_aten():
    out = dstack([torch.tensor(5.0), torch.tensor(6.0)])
    torch.testing.assert_close(
        out, torch.dstack([torch.tensor(5.0), torch.tensor(6.0)]), rtol=0, atol=0
    )
    a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    b = a * 10
    out2 = dstack([a, b])
    torch.testing.assert_close(out2, torch.dstack([a, b]), rtol=0, atol=0)


def test_dstack_3d_matches_aten():
    a = torch.arange(8, dtype=torch.float64).reshape(2, 2, 2)
    b = torch.arange(4, dtype=torch.float64).reshape(2, 2, 1)
    out = dstack([a, b])
    ref = torch.dstack([a, b])
    assert out.shape == ref.shape == (2, 2, 3)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    c = torch.arange(16, dtype=torch.float64).reshape(2, 2, 2, 2)
    d = torch.ones(2, 2, 2, 2, dtype=torch.float64)
    out2 = dstack([c, d])
    torch.testing.assert_close(out2, torch.dstack([c, d]), rtol=0, atol=0)


def test_dstack_mixed_rank_aligned():
    out = dstack([torch.arange(3), torch.arange(3).reshape(1, 3)])
    ref = torch.dstack([torch.arange(3), torch.arange(3).reshape(1, 3)])
    assert out.shape == (1, 3, 2)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_dstack_dtype_promotion():
    a = torch.arange(3, dtype=torch.float32)
    b = torch.arange(3, dtype=torch.float64)
    out = dstack([a, b])
    ref = torch.dstack([a, b])
    assert out.dtype == ref.dtype == torch.float64
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    c = torch.arange(3, dtype=torch.int64)
    out2 = dstack([a, c])
    assert out2.dtype == torch.float32
    torch.testing.assert_close(out2, torch.dstack([a, c]), rtol=0, atol=0)


def test_dstack_bool_and_non_contiguous():
    a = torch.tensor([True, False])
    b = torch.tensor([False, True])
    out = dstack([a, b])
    assert out.dtype == torch.bool
    torch.testing.assert_close(out, torch.dstack([a, b]), rtol=0, atol=0)
    nc = torch.arange(6, dtype=torch.float64).reshape(2, 3).t()
    assert not nc.is_contiguous()
    out2 = dstack([nc, nc])
    torch.testing.assert_close(out2, torch.dstack([nc, nc]), rtol=0, atol=0)


def test_dstack_out_variant():
    a = torch.arange(3, dtype=torch.float64)
    b = torch.arange(3, dtype=torch.float64) * 10
    out = torch.empty(1, 3, 2, dtype=torch.float64)
    ret = dstack([a, b], out=out)
    assert ret is out
    torch.testing.assert_close(out, torch.dstack([a, b]), rtol=0, atol=0)


def test_dstack_errors_match_aten():
    with pytest.raises(RuntimeError, match="dstack expects a non-empty TensorList"):
        dstack([])
    with pytest.raises(TypeError, match="must be tuple of Tensors, not Tensor"):
        dstack(torch.arange(3))
    with pytest.raises(TypeError, match="expected Tensor as element 1 in argument 0"):
        dstack([torch.arange(3), 5])
    with pytest.raises(
        RuntimeError,
        match=r"Sizes of tensors must match except in dimension 2\. "
        r"Expected size 2 but got size 3",
    ):
        dstack([torch.randn(2, 2), torch.randn(3, 2)])
    with pytest.raises(RuntimeError, match="same number of dimensions: got 3 and 4"):
        dstack([torch.zeros(2, 3, 4), torch.zeros(2, 3, 4, 5)])


def test_dstack_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.dstack")
    launches = []
    kernel = mod._dstack_copy_kernel
    monkeypatch.setattr(kernel, "run", _spy_on(kernel, launches))
    a = torch.arange(8, dtype=torch.float64).reshape(2, 2, 2)
    b = torch.arange(4, dtype=torch.float64).reshape(2, 2, 1)
    out = dstack([a, b])
    assert len(launches) == 2  # one launch per input
    torch.testing.assert_close(out, torch.dstack([a, b]), rtol=0, atol=0)
