import importlib

import pytest
import torch

from .spdiags import spdiags


@pytest.mark.parametrize(
    "offsets, shape",
    [
        ([0], (3, 3)),
        ([1], (3, 3)),
        ([-1], (3, 3)),
        ([2], (3, 3)),
        ([-2], (3, 3)),
        ([0, 1], (4, 4)),
        ([-1, 0, 2], (5, 4)),
        ([0, 1], (2, 3)),
        ([0, -1], (4, 4)),
        ([0, 1, -1], (5, 5)),
    ],
)
def test_spdiags_matches_torch(offsets, shape):
    # diag_len <= ncols keeps torch's own kernel out of its diag_len > ncols
    # heap-corruption zone (see test_spdiags_aten_bug_case_kept_correct).
    offs = torch.tensor(offsets, dtype=torch.int64)
    diag_len = min(4, shape[1])
    diagonals = torch.arange(len(offsets) * diag_len, dtype=torch.float64).reshape(
        len(offsets), diag_len
    )
    out = spdiags(diagonals, offs, shape)
    ref = torch.sparse.spdiags(diagonals, offs, shape)
    assert out.dtype == ref.dtype and out.shape == ref.shape
    assert out._nnz() == ref._nnz()
    torch.testing.assert_close(
        out.coalesce().indices(), ref.coalesce().indices(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        out.coalesce().values(), ref.coalesce().values(), rtol=0, atol=0
    )
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=0, atol=0)


def test_spdiags_long_diagonals_match_torch():
    # diag_len (5) > ncols (4): positions whose col lands outside are dropped,
    # exactly like torch (torch's own kernel is only buggy when a kept slot
    # has col >= ncols, which this shape avoids).
    diagonals = torch.arange(10.0, dtype=torch.float64).reshape(2, 5)
    offs = torch.tensor([0, 1], dtype=torch.int64)
    out = spdiags(diagonals, offs, (4, 4))
    ref = torch.sparse.spdiags(diagonals, offs, (4, 4))
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=0, atol=0)


def test_spdiags_aten_bug_case_kept_correct():
    # torch.sparse.spdiags(diag_len=3 -> (3, 2)) keeps an out-of-range column
    # (nnz 4) and corrupts the heap on to_dense(); the correct count is 3.
    diagonals = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    offs = torch.tensor([0, 1], dtype=torch.int64)
    out = spdiags(diagonals, offs, (3, 2))
    assert out._nnz() == 3
    dense = out.to_dense()
    expected = torch.zeros(3, 2, dtype=torch.float64)
    expected[0, 0] = 1.0
    expected[1, 1] = 2.0
    expected[0, 1] = 5.0
    torch.testing.assert_close(dense, expected, rtol=0, atol=0)


def test_spdiags_1d_diagonals_and_scalar_offset_promotion():
    d1 = torch.tensor([10.0, 20.0, 30.0])
    out = spdiags(d1, torch.tensor(0, dtype=torch.int64), (3, 3))
    ref = torch.sparse.spdiags(d1, torch.tensor(0, dtype=torch.int64), (3, 3))
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=0, atol=0)
    assert out._nnz() == 3


def test_spdiags_negative_offsets_rectangular():
    for off in (-3, -2, 2, 3):
        offs = torch.tensor([off], dtype=torch.int64)
        diagonals = torch.arange(3, dtype=torch.float32)
        out = spdiags(diagonals, offs, (4, 3))
        ref = torch.sparse.spdiags(diagonals, offs, (4, 3))
        torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=0, atol=0)


def test_spdiags_dtypes():
    di = torch.arange(6, dtype=torch.int64).reshape(2, 3)
    offs = torch.tensor([0, 1], dtype=torch.int64)
    out = spdiags(di, offs, (3, 3))
    ref = torch.sparse.spdiags(di, offs, (3, 3))
    assert out.dtype == torch.int64
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=0, atol=0)


def test_spdiags_layouts():
    diagonals = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    offs = torch.tensor([0, 1], dtype=torch.int64)
    for layout in (None, torch.sparse_coo, torch.sparse_csr, torch.sparse_csc):
        out = spdiags(diagonals, offs, (3, 3), layout=layout)
        ref = torch.sparse.spdiags(
            diagonals, offs, (3, 3), layout=layout if layout else torch.sparse_coo
        )
        expected_layout = layout if layout else torch.sparse_coo
        assert out.layout == expected_layout
        torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=0, atol=0)


def test_spdiags_empty_diagonals():
    se = spdiags(torch.empty((0, 3)), torch.empty(0, dtype=torch.int64), (3, 3))
    assert se.shape == (3, 3) and se._nnz() == 0


def test_spdiags_errors_match_torch_messages():
    diagonals = torch.arange(6.0).reshape(2, 3)
    with pytest.raises(RuntimeError, match="Number of diagonals"):
        spdiags(diagonals, torch.tensor([0], dtype=torch.int64), (3, 3))
    with pytest.raises(RuntimeError, match="duplicate values"):
        spdiags(diagonals, torch.tensor([0, 0], dtype=torch.int64), (3, 3))
    with pytest.raises(RuntimeError, match="must have dtype Long"):
        spdiags(diagonals, torch.tensor([0, 1], dtype=torch.int32), (3, 3))
    with pytest.raises(RuntimeError, match="Output shape must be 2d"):
        spdiags(diagonals, torch.tensor([0, 1], dtype=torch.int64), (3,))
    with pytest.raises(RuntimeError, match="Diagonals must be vector or matrix"):
        spdiags(
            torch.randn(2, 2, 2), torch.tensor([0, 1, 2, 3], dtype=torch.int64), (3, 3)
        )
    with pytest.raises(RuntimeError, match="Only output layouts"):
        spdiags(
            diagonals,
            torch.tensor([0, 1], dtype=torch.int64),
            (3, 3),
            layout=torch.strided,
        )


def test_spdiags_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.spdiags")
    launches = []
    kernel = mod.spdiags_kernel
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    diagonals = torch.arange(8.0, dtype=torch.float64).reshape(2, 4)
    offs = torch.tensor([-1, 2], dtype=torch.int64)
    out = spdiags(diagonals, offs, (4, 4))
    ref = torch.sparse.spdiags(diagonals, offs, (4, 4))
    assert len(launches) >= 1
    torch.testing.assert_close(out.to_dense(), ref.to_dense(), rtol=0, atol=0)
    torch.testing.assert_close(
        out.coalesce().indices(), ref.coalesce().indices(), rtol=0, atol=0
    )
