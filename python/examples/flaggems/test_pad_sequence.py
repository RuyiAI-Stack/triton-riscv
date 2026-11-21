import importlib

import pytest
import torch

from torch.nn.utils.rnn import pad_sequence as aten_pad_sequence

from .pad_sequence import pad_sequence


def _spy_on(kernel, launches):
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    return spy_run


def test_pad_sequence_both_layouts_match_aten():
    torch.manual_seed(0)
    a = torch.randn(3, 2)
    b = torch.randn(5, 2)
    for batch_first in (False, True):
        out = pad_sequence([a, b], batch_first=batch_first)
        ref = aten_pad_sequence([a, b], batch_first=batch_first)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        assert out.shape == ref.shape


def test_pad_sequence_padding_value():
    out = pad_sequence([torch.tensor([1, 2]), torch.tensor([3])], padding_value=-1.0)
    ref = aten_pad_sequence(
        [torch.tensor([1, 2]), torch.tensor([3])], padding_value=-1.0
    )
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    out_i = pad_sequence([torch.tensor([1, 2]), torch.tensor([3])], padding_value=7)
    ref_i = aten_pad_sequence(
        [torch.tensor([1, 2]), torch.tensor([3])], padding_value=7
    )
    torch.testing.assert_close(out_i, ref_i, rtol=0, atol=0)
    # float padding into an int tensor truncates exactly like aten
    out_f = pad_sequence([torch.tensor([1, 2]), torch.tensor([3])], padding_value=2.5)
    ref_f = aten_pad_sequence(
        [torch.tensor([1, 2]), torch.tensor([3])], padding_value=2.5
    )
    assert out_f.dtype == torch.int64
    torch.testing.assert_close(out_f, ref_f, rtol=0, atol=0)


def test_pad_sequence_empty_sequences_in_list():
    torch.manual_seed(1)
    seqs = [torch.empty(0, 2), torch.randn(2, 2)]
    out = pad_sequence(seqs)
    ref = aten_pad_sequence(seqs)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    all_empty = [torch.empty(0), torch.empty(0)]
    out2 = pad_sequence(all_empty)
    ref2 = aten_pad_sequence(all_empty)
    assert out2.shape == ref2.shape == (0, 2)
    single_empty = pad_sequence([torch.empty(0)])
    assert single_empty.shape == (0, 1)


def test_pad_sequence_single_sequence():
    torch.manual_seed(2)
    s = torch.randn(3, 4)
    for batch_first in (False, True):
        out = pad_sequence([s], batch_first=batch_first)
        ref = aten_pad_sequence([s], batch_first=batch_first)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_pad_sequence_3d_trailing():
    torch.manual_seed(3)
    seqs = [torch.randn(2, 3, 4), torch.randn(4, 3, 4)]
    out = pad_sequence(seqs, batch_first=True)
    ref = aten_pad_sequence(seqs, batch_first=True)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_pad_sequence_dispatch_paths_match_aten():
    torch.manual_seed(4)
    # batch 3..8, small total -> direct-copy host path
    seqs = [torch.randn(3, 2) for _ in range(3)]
    torch.testing.assert_close(
        pad_sequence(seqs), aten_pad_sequence(seqs), rtol=0, atol=0
    )
    # batch 5, batch_first -> direct-copy host path
    seqs_bf = [torch.randn(2, 5) for _ in range(5)]
    torch.testing.assert_close(
        pad_sequence(seqs_bf, batch_first=True),
        aten_pad_sequence(seqs_bf, batch_first=True),
        rtol=0,
        atol=0,
    )
    # batch 3, total > 8192, not batch_first -> batch-copy kernel path
    big = [torch.randn(40, 80) for _ in range(3)]
    torch.testing.assert_close(
        pad_sequence(big), aten_pad_sequence(big), rtol=0, atol=0
    )
    # batch > 8 -> flat kernel path, mixed lengths
    many = [torch.randn(n, 3) for n in range(1, 12)]
    torch.testing.assert_close(
        pad_sequence(many), aten_pad_sequence(many), rtol=0, atol=0
    )
    many_bf = [torch.randn(n, 3) for n in range(1, 12)]
    torch.testing.assert_close(
        pad_sequence(many_bf, batch_first=True),
        aten_pad_sequence(many_bf, batch_first=True),
        rtol=0,
        atol=0,
    )


def test_pad_sequence_mixed_dtype_uses_first_dtype():
    a = torch.arange(3, dtype=torch.int64)
    b = torch.randn(5)
    out = pad_sequence([a, b])
    ref = aten_pad_sequence([a, b])
    assert out.dtype == ref.dtype == torch.int64
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    out2 = pad_sequence([b, a])
    ref2 = aten_pad_sequence([b, a])
    assert out2.dtype == ref2.dtype
    torch.testing.assert_close(out2, ref2, rtol=0, atol=0)


def test_pad_sequence_non_contiguous_input():
    torch.manual_seed(5)
    s = torch.randn(4, 6).t()
    other = torch.randn(3, 4)
    assert not s.is_contiguous()
    out = pad_sequence([s, other])
    ref = aten_pad_sequence([s, other])
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_pad_sequence_errors_match_aten():
    with pytest.raises(RuntimeError, match="received an empty list of sequences"):
        pad_sequence([])
    with pytest.raises(RuntimeError, match="ArrayRef: invalid slice"):
        pad_sequence([torch.tensor(5.0)])
    with pytest.raises(
        RuntimeError,
        match=r"The size of tensor a \(2\) must match the size of tensor b \(5\) "
        r"at non-singleton dimension 1",
    ):
        pad_sequence([torch.randn(3, 2), torch.randn(3, 5)])
    with pytest.raises(
        RuntimeError,
        match=r"The size of tensor a \(5\) must match the size of tensor b \(2\) "
        r"at non-singleton dimension 1",
    ):
        pad_sequence([torch.randn(3, 5), torch.randn(3, 2)])
    with pytest.raises(
        RuntimeError,
        match=r"The size of tensor a \(4\) must match the size of tensor b \(5\) "
        r"at non-singleton dimension 2",
    ):
        pad_sequence([torch.randn(3, 2, 4), torch.randn(3, 2, 5)])


def test_pad_sequence_triton_path_small_kernel(monkeypatch):
    mod = importlib.import_module("flaggems.pad_sequence")
    launches = []
    kernel = mod._pad_sequence_small_kernel
    monkeypatch.setattr(kernel, "run", _spy_on(kernel, launches))
    a = torch.randn(3, 2)
    b = torch.randn(5, 2)
    out = pad_sequence([a, b])
    assert len(launches) == 2  # one launch per sequence on this path
    ref = aten_pad_sequence([a, b])
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    # batch 0 has length 3 of max 5: its tail rows are padding
    assert torch.all(out[3:, 0] == 0)


def test_pad_sequence_triton_path_batch_copy_kernel(monkeypatch):
    mod = importlib.import_module("flaggems.pad_sequence")
    launches = []
    kernel = mod._pad_sequence_batch_copy_kernel
    monkeypatch.setattr(kernel, "run", _spy_on(kernel, launches))
    seqs = [torch.randn(40, 80) for _ in range(3)]  # total 9600 > 8192
    out = pad_sequence(seqs)
    assert len(launches) >= 1
    torch.testing.assert_close(out, aten_pad_sequence(seqs), rtol=0, atol=0)


def test_pad_sequence_triton_path_flat_kernel(monkeypatch):
    mod = importlib.import_module("flaggems.pad_sequence")
    launches = []
    kernel = mod._pad_sequence_flat_kernel
    monkeypatch.setattr(kernel, "run", _spy_on(kernel, launches))
    seqs = [torch.randn(n, 3) for n in range(1, 12)]
    out = pad_sequence(seqs, batch_first=True, padding_value=9.0)
    assert len(launches) >= 1
    torch.testing.assert_close(
        out,
        aten_pad_sequence(seqs, batch_first=True, padding_value=9.0),
        rtol=0,
        atol=0,
    )
