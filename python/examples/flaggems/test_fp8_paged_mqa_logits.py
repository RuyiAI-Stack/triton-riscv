import pytest
import torch

from .fp8_paged_mqa_logits import fp8_paged_mqa_logits


@pytest.mark.parametrize(
    "batch_size, next_n, heads, dim, num_blocks, block_size",
    [
        (1, 2, 2, 8, 4, 4),
    ],
)
def test_fp8_paged_mqa_logits(batch_size, next_n, heads, dim, num_blocks, block_size):
    torch.manual_seed(0)
    q = torch.randn(batch_size, next_n, heads, dim, dtype=torch.float32)
    kv_fp8 = torch.randn(num_blocks, block_size, dim).to(torch.float8_e4m3fn)
    scales = torch.linspace(0.5, 1.5, num_blocks * block_size).reshape(
        num_blocks, block_size
    )
    kv = torch.empty((num_blocks, block_size, 1, dim + 4), dtype=torch.uint8)
    kv[:, :, 0, :dim] = kv_fp8.view(torch.uint8)
    kv[:, :, 0, dim:] = scales.unsqueeze(-1).contiguous().view(torch.uint8)
    weights = torch.randn(batch_size * next_n, heads, dtype=torch.float32)
    context_lens = torch.full((batch_size,), next_n, dtype=torch.int32)
    block_order = torch.roll(torch.arange(num_blocks, dtype=torch.int32), shifts=1)
    block_tables = block_order.repeat(batch_size, 1)
    max_model_len = num_blocks * block_size

    tri = fp8_paged_mqa_logits(
        q, kv, weights, context_lens, block_tables, max_model_len
    )

    logical_positions = torch.arange(next_n)
    logical_blocks = logical_positions // block_size
    intra_block_positions = logical_positions % block_size
    ref = torch.full_like(tri, float("-inf"))
    for batch_idx in range(batch_size):
        physical_blocks = block_tables[batch_idx, logical_blocks].long()
        keys = kv_fp8.float()[physical_blocks, intra_block_positions]
        keys *= scales[physical_blocks, intra_block_positions, None]
        for query_idx in range(next_n):
            row = batch_idx * next_n + query_idx
            scores = torch.einsum(
                "hd,nd->hn", q[batch_idx, query_idx], keys[: query_idx + 1]
            )
            ref[row, : query_idx + 1] = torch.sum(
                torch.relu(scores) * weights[row, :, None], dim=0
            )

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
