import torch
import triton
import triton.language as tl


@triton.jit
def _paged_mqa_logits_metadata_kernel(
    context_lens_ptr,
    context_lens_stride,
    schedule_metadata_ptr,
    batch_size,
    split_kv,
    num_sms,
    BLOCK_SIZE: tl.constexpr,
):
    sm_idx = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

    ctx_lens = tl.load(
        context_lens_ptr + offsets * context_lens_stride, mask=mask, other=0
    )

    num_segs = (ctx_lens + split_kv - 1) // split_kv
    num_segs = tl.where(mask, num_segs, 0)

    prefix_sum = tl.cumsum(num_segs, axis=0)

    total_segs = tl.max(prefix_sum)

    q = total_segs // num_sms
    r = total_segs % num_sms
    min_r = sm_idx if sm_idx < r else r
    seg_starts = sm_idx * q + min_r

    is_le = (prefix_sum <= seg_starts) & mask
    q_idx = tl.sum(tl.where(is_le, 1, 0))

    prev_mask = offsets < q_idx
    prev_prefix = tl.max(tl.where(prev_mask, prefix_sum, 0))
    kv_split_idx = seg_starts - prev_prefix

    out_idx = sm_idx * 2
    tl.store(schedule_metadata_ptr + out_idx, q_idx)
    tl.store(schedule_metadata_ptr + out_idx + 1, kv_split_idx)


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor, block_size: int, num_sms: int
) -> torch.Tensor:
    SPLIT_KV = 256
    device = context_lens.device

    if context_lens.dim() == 2:
        batch_size, next_n = context_lens.shape
        effective_context_lens = context_lens[:, next_n - 1]
    else:
        batch_size = context_lens.shape[0]
        effective_context_lens = context_lens

    if batch_size == 0:
        return torch.zeros((num_sms + 1, 2), dtype=torch.int32, device=device)

    grid = (num_sms + 1,)

    BLOCK_SIZE = triton.next_power_of_2(max(16, batch_size))

    schedule_metadata = torch.zeros((num_sms + 1, 2), dtype=torch.int32, device=device)

    _paged_mqa_logits_metadata_kernel[grid](
        effective_context_lens,
        effective_context_lens.stride(0),
        schedule_metadata,
        batch_size,
        SPLIT_KV,
        num_sms,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return schedule_metadata
