import pytest
import torch

from .fp8_paged_mqa_logits import fp8_paged_mqa_logits


@pytest.mark.parametrize(
    "batch_size, next_n, heads, dim, num_blocks, block_size",
    [
        (1, 2, 2, 8, 4, 4),
    ],
)
def test_fp8_paged_mqa_logits(
    batch_size, next_n, heads, dim, num_blocks, block_size
):
    torch.manual_seed(0)
    q = torch.randn(batch_size, next_n, heads, dim, dtype=torch.float32)
    kv = torch.randint(
        0, 64, (num_blocks, block_size, 1, dim + 4), dtype=torch.uint8
    )
    weights = torch.randn(batch_size * next_n, heads, dtype=torch.float32)
    context_lens = torch.full((batch_size,), next_n, dtype=torch.int32)
    block_tables = torch.arange(num_blocks, dtype=torch.int32).reshape(
        batch_size, num_blocks
    )
    max_model_len = num_blocks * block_size

    tri = fp8_paged_mqa_logits(
        q, kv, weights, context_lens, block_tables, max_model_len
    )

    assert tri.shape == (batch_size * next_n, max_model_len)
    assert tri.dtype == torch.float32
