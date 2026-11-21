import importlib

import pytest
import torch

from ._pad_packed_sequence import _pad_packed_sequence


def _aten(data, bs, bf=False, pv=0.0, tl=-1):
    return torch.ops.aten._pad_packed_sequence(data, bs, bf, pv, tl)


@pytest.mark.parametrize("batch_first", [False, True])
def test_pps_matches_aten(batch_first):
    torch.manual_seed(0)
    data = torch.randn(7, 2)
    bs = torch.tensor([3, 2, 2])
    out, lens = _pad_packed_sequence(data, bs, batch_first, 0.0, -1)
    ref_out, ref_lens = _aten(data, bs, batch_first)
    assert out.shape == ref_out.shape
    assert lens.tolist() == ref_lens.tolist() == [3, 3, 1]
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)


def test_pps_total_length_and_padding_value():
    torch.manual_seed(0)
    data = torch.randn(7, 2)
    bs = torch.tensor([3, 2, 2])
    out, lens = _pad_packed_sequence(data, bs, False, 9.0, 5)
    ref_out, ref_lens = _aten(data, bs, False, 9.0, 5)
    assert out.shape == (5, 3, 2)
    assert lens.tolist() == ref_lens.tolist()
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    assert (out[3:] == 9.0).all()  # the padded tail rows


def test_pps_multi_dim_feature():
    torch.manual_seed(0)
    data = torch.randn(7, 2, 3)
    bs = torch.tensor([3, 2, 2])
    out, lens = _pad_packed_sequence(data, bs, False, 0.0, -1)
    ref_out, ref_lens = _aten(data, bs, False)
    assert out.shape == (3, 3, 2, 3)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)


def test_pps_single_step_and_single_batch():
    data = torch.randn(4, 1)
    bs = torch.tensor([4])
    out, lens = _pad_packed_sequence(data, bs, False, -2.0, -1)
    ref_out, ref_lens = _aten(data, bs, False, -2.0, -1)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    # one time step active for every batch element
    assert lens.tolist() == ref_lens.tolist() == [1, 1, 1, 1]
    data2 = torch.randn(3, 2)
    bs2 = torch.tensor([2, 1])
    out2, lens2 = _pad_packed_sequence(data2, bs2, True, 0.0, -1)
    ref2, rl2 = _aten(data2, bs2, True)
    torch.testing.assert_close(out2, ref2, rtol=0, atol=0)
    assert lens2.tolist() == rl2.tolist() == [2, 1]


def test_pps_equal_lengths():
    torch.manual_seed(0)
    data = torch.randn(6, 2)
    bs = torch.tensor([2, 2, 2])
    out, lens = _pad_packed_sequence(data, bs, False, 0.0, -1)
    ref, rl = _aten(data, bs, False)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert lens.tolist() == [3, 3]


def test_pps_int_data_int_padding():
    data = torch.arange(9, dtype=torch.int64).reshape(9, 1)
    bs = torch.tensor([3, 3])
    out, lens = _pad_packed_sequence(data, bs, False, 7, -1)
    ref, rl = _aten(data, bs, False, 7, -1)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert lens.tolist() == rl.tolist()


def test_pps_fp64_exact():
    torch.manual_seed(0)
    data = torch.randn(12, 3, dtype=torch.float64)
    bs = torch.tensor([4, 3, 3, 2])
    out, lens = _pad_packed_sequence(data, bs, False, 1.5, 6)
    ref, rl = _aten(data, bs, False, 1.5, 6)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert lens.tolist() == rl.tolist()


def test_pps_rnn_roundtrip():
    # Cross-check against torch.nn.utils.rnn on a real packed batch.
    torch.manual_seed(0)
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    seq = torch.randn(3, 5, dtype=torch.float64)
    lens_t = torch.tensor([5, 3, 1])
    packed = pack_padded_sequence(seq, lens_t, batch_first=True, enforce_sorted=True)
    out, lens = _pad_packed_sequence(packed.data, packed.batch_sizes, True, 0.0, -1)
    un, ul = pad_packed_sequence(packed, batch_first=True)
    torch.testing.assert_close(out, un, rtol=0, atol=0)
    assert lens.tolist() == ul.tolist()


def test_pps_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._pad_packed_sequence")
    launches = []
    kernel = mod._pad_packed_sequence_kernel
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(0)
    data = torch.randn(7, 2)
    bs = torch.tensor([3, 2, 2])
    out, lens = _pad_packed_sequence(data, bs, False, 4.0, 4)
    ref_out, ref_lens = _aten(data, bs, False, 4.0, 4)
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    assert lens.tolist() == ref_lens.tolist()
