import importlib

import pytest
import torch

from .block_diag import block_diag


def test_block_diag_uniform_2d_matches_aten():
    a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    b = torch.arange(4, dtype=torch.float64).reshape(2, 2) * 10
    c = torch.ones(1, 1, dtype=torch.float64)
    out = block_diag(a, b, c)
    ref = torch.block_diag(a, b, c)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == ref.shape


def test_block_diag_mixed_ranks_matches_aten():
    out = block_diag(torch.tensor(9), torch.arange(2), torch.ones(2, 2))
    ref = torch.block_diag(torch.tensor(9), torch.arange(2), torch.ones(2, 2))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_block_diag_1d_only_matches_aten():
    out = block_diag(torch.arange(3), torch.arange(2))
    ref = torch.block_diag(torch.arange(3), torch.arange(2))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_block_diag_dtype_promotion():
    a = torch.arange(2, dtype=torch.float32)
    b = torch.arange(2, dtype=torch.int64)
    out = block_diag(a, b)
    ref = torch.block_diag(a, b)
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    out2 = block_diag(b, a)
    ref2 = torch.block_diag(b, a)
    assert out2.dtype == ref2.dtype
    torch.testing.assert_close(out2, ref2, rtol=0, atol=0)


def test_block_diag_single_inputs():
    for t in (
        torch.arange(4).reshape(2, 2),
        torch.arange(3),
        torch.tensor(7.0),
    ):
        out = block_diag(t)
        ref = torch.block_diag(t)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        assert out.dtype == ref.dtype


def test_block_diag_bool_dtype():
    out = block_diag(torch.tensor([True, False]), torch.tensor([[False, True]]))
    ref = torch.block_diag(torch.tensor([True, False]), torch.tensor([[False, True]]))
    assert out.dtype == torch.bool
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_block_diag_non_contiguous_input():
    x = torch.arange(4, dtype=torch.float64).reshape(2, 2).t()
    assert not x.is_contiguous()
    out = block_diag(x)
    ref = torch.block_diag(x)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_block_diag_empty_blocks_and_empty_call():
    out = block_diag(torch.zeros(2, 0), torch.ones(1, 1))
    ref = torch.block_diag(torch.zeros(2, 0), torch.ones(1, 1))
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    empty = block_diag()
    ref_empty = torch.block_diag()
    assert empty.shape == ref_empty.shape  # (1, 0), unlike upstream's (0,)
    assert empty.dtype == ref_empty.dtype


def test_block_diag_errors_match_aten():
    with pytest.raises(RuntimeError, match="must have 2 or fewer dimensions"):
        block_diag(torch.ones(2, 2), torch.ones(2, 2, 2))
    with pytest.raises(
        TypeError, match="expected Tensor as element 0 in argument 0, but got list"
    ):
        block_diag([torch.ones(2, 2), torch.ones(2, 2)])
    with pytest.raises(
        TypeError, match="expected Tensor as element 1 in argument 0, but got int"
    ):
        block_diag(torch.tensor(3.0), 5)


def test_block_diag_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.block_diag")
    launches = []
    kernel = mod._block_diag_copy_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    b = torch.arange(4, dtype=torch.float64).reshape(2, 2) * 10
    out = block_diag(a, b)
    assert len(launches) == 2  # one launch per input tensor
    torch.testing.assert_close(out, torch.block_diag(a, b), rtol=0, atol=0)
