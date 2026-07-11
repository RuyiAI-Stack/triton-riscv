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
):
    sm_idx = tl.program_id(0)
    total_segs = tl.full((), 0, dtype=tl.int32)
    for batch_idx in range(0, batch_size):
        ctx_len = tl.load(
            context_lens_ptr + batch_idx * context_lens_stride
        ).to(tl.int32)
        total_segs += (ctx_len + split_kv - 1) // split_kv

    q = total_segs // num_sms
    r = total_segs % num_sms
    min_r = sm_idx if sm_idx < r else r
    seg_starts = sm_idx * q + min_r

    prefix_sum = tl.full((), 0, dtype=tl.int32)
    q_idx = tl.full((), 0, dtype=tl.int32)
    prev_prefix = tl.full((), 0, dtype=tl.int32)
    for batch_idx in range(0, batch_size):
        ctx_len = tl.load(
            context_lens_ptr + batch_idx * context_lens_stride
        ).to(tl.int32)
        prefix_sum += (ctx_len + split_kv - 1) // split_kv
        belongs_to_previous_query = prefix_sum <= seg_starts
        q_idx = tl.where(belongs_to_previous_query, batch_idx + 1, q_idx)
        prev_prefix = tl.where(
            belongs_to_previous_query, prefix_sum, prev_prefix
        )
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

    schedule_metadata = torch.zeros(
        (num_sms + 1, 2), dtype=torch.int32, device=device
    )

    _paged_mqa_logits_metadata_kernel[grid](
        effective_context_lens,
        effective_context_lens.stride(0),
        schedule_metadata,
        batch_size,
        SPLIT_KV,
        num_sms,
    )

    return schedule_metadata
