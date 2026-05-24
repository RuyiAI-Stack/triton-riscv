import pytest
import torch

from .get_paged_mqa_logits_metadata import get_paged_mqa_logits_metadata


def _ref_metadata(context_lens, block_size, num_sms):
    """Reference implementation matching the Triton kernel logic."""
    SPLIT_KV = 256
    if context_lens.dim() == 2:
        effective = context_lens[:, -1]
    else:
        effective = context_lens

    batch_size = effective.shape[0]
    if batch_size == 0:
        return torch.zeros(
            (num_sms + 1, 2), dtype=torch.int32, device=effective.device
        )

    num_segs = (effective + SPLIT_KV - 1) // SPLIT_KV
    prefix_sum = num_segs.cumsum(0)
    total_segs = prefix_sum[-1].item()

    q = total_segs // num_sms
    r = total_segs % num_sms

    metadata = torch.zeros(
        (num_sms + 1, 2), dtype=torch.int32, device=effective.device
    )
    for sm_idx in range(num_sms + 1):
        min_r = sm_idx if sm_idx < r else r
        seg_starts = sm_idx * q + min_r

        is_le = prefix_sum <= seg_starts
        q_idx = is_le.sum().item()

        prev_mask = torch.arange(batch_size) < q_idx
        prev_prefix = (
            prefix_sum[prev_mask].max().item() if prev_mask.any() else 0
        )
        kv_split_idx = seg_starts - prev_prefix

        metadata[sm_idx, 0] = q_idx
        metadata[sm_idx, 1] = kv_split_idx

    return metadata


@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
def test_paged_mqa_logits_metadata_1d(batch_size):
    torch.manual_seed(0)
    context_lens = torch.randint(
        1, 4096, (batch_size,), dtype=torch.int32, device="cpu"
    )
    num_sms = 8
    block_size = 256

    ref = _ref_metadata(context_lens, block_size, num_sms)
    tri = get_paged_mqa_logits_metadata(context_lens, block_size, num_sms)

    assert tri.shape == ref.shape
    assert tri.dtype == ref.dtype
    assert torch.equal(tri, ref), f"Mismatch for batch_size={batch_size}"


def test_paged_mqa_logits_metadata_2d():
    torch.manual_seed(0)
    batch_size, next_n = 8, 4
    context_lens = torch.randint(
        1, 4096, (batch_size, next_n), dtype=torch.int32, device="cpu"
    )
    num_sms = 8
    block_size = 256

    tri = get_paged_mqa_logits_metadata(context_lens, block_size, num_sms)
    ref = _ref_metadata(context_lens, block_size, num_sms)

    assert torch.equal(tri, ref)


def test_paged_mqa_logits_metadata_empty_batch():
    context_lens = torch.empty(0, dtype=torch.int32, device="cpu")
    num_sms = 8
    block_size = 256
    result = get_paged_mqa_logits_metadata(context_lens, block_size, num_sms)
    assert result.shape == (num_sms + 1, 2)
    assert (result == 0).all()
