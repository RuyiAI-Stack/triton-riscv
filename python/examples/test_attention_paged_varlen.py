import math

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


def make_paged_inputs_case_one():
    torch.manual_seed(2)
    query_lens = torch.tensor([2, 3], dtype=torch.int32)
    kv_lens = torch.tensor([4, 5], dtype=torch.int32)
    query = torch.randn((5, 2, 8), device="cpu", dtype=torch.float32)
    key_cache = torch.randn((4, 4, 2, 8), device="cpu", dtype=torch.float32)
    value_cache = torch.randn((4, 4, 2, 8), device="cpu", dtype=torch.float32)
    block_table = torch.tensor([[0, -1], [2, 3]], dtype=torch.int32)
    return (
        query.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        query_lens,
        kv_lens,
        block_table,
    )


def make_paged_inputs_case_two():
    torch.manual_seed(3)
    query_lens = torch.tensor([1, 2], dtype=torch.int32)
    kv_lens = torch.tensor([3, 5], dtype=torch.int32)
    query = torch.randn((3, 2, 10), device="cpu", dtype=torch.float32)
    key_cache = torch.randn((4, 4, 2, 10), device="cpu", dtype=torch.float32)
    value_cache = torch.randn((4, 4, 2, 10), device="cpu", dtype=torch.float32)
    block_table = torch.tensor([[1, 0], [2, 3]], dtype=torch.int32)
    return (
        query.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        query_lens,
        kv_lens,
        block_table,
    )


def _validate_paged_attention_inputs(
    query,
    key_cache,
    value_cache,
    query_lens,
    kv_lens,
    block_table,
):
    tensors = {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "query_lens": query_lens,
        "kv_lens": kv_lens,
        "block_table": block_table,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")

    for name, tensor in (
        ("query", query),
        ("key_cache", key_cache),
        ("value_cache", value_cache),
    ):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32")

    for name, tensor in (("query_lens", query_lens), ("kv_lens", kv_lens), ("block_table", block_table)):
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must have dtype torch.int32")

    for name, tensor in (
        ("query", query),
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("query_lens", query_lens),
        ("kv_lens", kv_lens),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if query.ndim != 3:
        raise ValueError("query must be 3D")
    if key_cache.ndim != 4:
        raise ValueError("key_cache must be 4D")
    if value_cache.ndim != 4:
        raise ValueError("value_cache must be 4D")
    if block_table.ndim != 2:
        raise ValueError("block_table must be 2D")
    if query.shape[1] != key_cache.shape[2] or query.shape[1] != value_cache.shape[2]:
        raise ValueError("query, key_cache, and value_cache must agree on num_heads")
    if query.shape[2] != key_cache.shape[3] or query.shape[2] != value_cache.shape[3]:
        raise ValueError("query, key_cache, and value_cache must agree on head_dim")
    if key_cache.shape[0] != value_cache.shape[0]:
        raise ValueError("key_cache and value_cache must have the same number of blocks")
    if key_cache.shape[1] != value_cache.shape[1]:
        raise ValueError("key_cache and value_cache must have the same block size")
    if int(query_lens.sum().item()) != query.shape[0]:
        raise ValueError("query_lens must sum to the total number of query tokens")
    if query_lens.shape != kv_lens.shape:
        raise ValueError("query_lens and kv_lens must have the same shape")
    if block_table.shape[0] != query_lens.shape[0]:
        raise ValueError("block_table must have one row per sequence")

    block_size = key_cache.shape[1]
    num_cache_blocks = key_cache.shape[0]
    max_blocks_per_seq = block_table.shape[1]

    for seq_idx, (q_len, kv_len) in enumerate(zip(query_lens.tolist(), kv_lens.tolist())):
        if q_len > kv_len:
            raise ValueError("each query sequence must be no longer than its KV sequence")
        required_blocks = (kv_len + block_size - 1) // block_size
        if required_blocks > max_blocks_per_seq:
            raise ValueError("block_table does not provide enough blocks for kv_len")
        if required_blocks == 0:
            continue
        required_block_ids = block_table[seq_idx, :required_blocks]
        if not torch.all(required_block_ids >= 0):
            raise ValueError("block_table entries must be non-negative for used blocks")
        if not torch.all(required_block_ids < num_cache_blocks):
            raise ValueError("block_table entries must reference valid cache blocks")

    return block_size, max_blocks_per_seq


def attention_paged_varlen_reference(
    query,
    key_cache,
    value_cache,
    query_lens,
    kv_lens,
    block_table,
):
    """Reference for suffix-aligned paged varlen attention.

    Each query sequence is aligned to the final `q_len` positions of its KV
    sequence. Query token `j` may attend only through KV position
    `kv_len - q_len + j`. This example is intentionally not a generic packed
    causal attention implementation.
    """
    block_size, _ = _validate_paged_attention_inputs(
        query, key_cache, value_cache, query_lens, kv_lens, block_table
    )
    head_dim = query.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    outputs = []
    cursor = 0

    for seq_idx, (q_len, kv_len) in enumerate(zip(query_lens.tolist(), kv_lens.tolist())):
        q_seq = query[cursor : cursor + q_len]
        num_blocks = (kv_len + block_size - 1) // block_size
        block_ids = block_table[seq_idx, :num_blocks].tolist()
        k_seq = torch.cat([key_cache[block_id] for block_id in block_ids], dim=0)[:kv_len]
        v_seq = torch.cat([value_cache[block_id] for block_id in block_ids], dim=0)[:kv_len]

        seq_out = torch.empty((q_len, query.shape[1], head_dim), device="cpu", dtype=torch.float32)
        # The query block is anchored to the suffix of the KV sequence.
        base_k = kv_len - q_len
        kv_positions = torch.arange(kv_len)

        for q_idx in range(q_len):
            scores = torch.einsum("hd,khd->hk", q_seq[q_idx], k_seq) * scale
            mask = kv_positions > (base_k + q_idx)
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            seq_out[q_idx] = torch.einsum("hk,khd->hd", attn, v_seq)

        outputs.append(seq_out)
        cursor += q_len

    return torch.cat(outputs, dim=0)


def build_query_metadata(query_lens):
    total_q = int(query_lens.sum().item())
    query_seq_ids = torch.empty((total_q,), device="cpu", dtype=torch.int32)
    query_offsets = torch.empty((total_q,), device="cpu", dtype=torch.int32)

    cursor = 0
    for seq_idx, q_len in enumerate(query_lens.tolist()):
        query_seq_ids[cursor : cursor + q_len] = seq_idx
        query_offsets[cursor : cursor + q_len] = torch.arange(q_len, dtype=torch.int32)
        cursor += q_len

    return query_seq_ids.contiguous(), query_offsets.contiguous()


@triton.jit
def _paged_varlen_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    out_ptr,
    query_seq_ids_ptr,
    query_offsets_ptr,
    query_lens_ptr,
    kv_lens_ptr,
    block_table_ptr,
    total_q,
    num_heads,
    head_dim,
    block_size,
    max_blocks_per_seq,
    max_kv_len,
    stride_bts,
    stride_btc,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    scale,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    token_idx = pid // num_heads
    head_idx = pid % num_heads
    offs_d = tl.arange(0, BLOCK_D)

    seq_idx = tl.load(query_seq_ids_ptr + token_idx)
    query_offset = tl.load(query_offsets_ptr + token_idx)
    query_len = tl.load(query_lens_ptr + seq_idx)
    kv_len = tl.load(kv_lens_ptr + seq_idx)
    # Align each query position to the suffix of its KV sequence.
    max_allowed_k = kv_len - query_len + query_offset

    q = tl.load(
        query_ptr + token_idx * stride_qt + head_idx * stride_qh + offs_d * stride_qd,
        mask=offs_d < head_dim,
        other=0.0,
    ).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for logical_k in range(0, max_kv_len):
        safe_logical_k = tl.where(logical_k < kv_len, logical_k, 0)
        table_col = safe_logical_k // block_size
        block_off = safe_logical_k % block_size
        block_idx = tl.load(block_table_ptr + seq_idx * stride_bts + table_col * stride_btc)

        k = tl.load(
            key_cache_ptr
            + block_idx * stride_kb
            + block_off * stride_kt
            + head_idx * stride_kh
            + offs_d * stride_kd,
            mask=offs_d < head_dim,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            value_cache_ptr
            + block_idx * stride_vb
            + block_off * stride_vt
            + head_idx * stride_vh
            + offs_d * stride_vd,
            mask=offs_d < head_dim,
            other=0.0,
        ).to(tl.float32)

        score = tl.sum(q * k, axis=0) * scale
        valid = (logical_k < kv_len) & (logical_k <= max_allowed_k)
        score = tl.where(valid, score, -float("inf"))

        m_new = tl.maximum(m_i, score)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(score - m_new)

        acc = acc * alpha + p * v
        l_i = l_i * alpha + p
        m_i = m_new

    out = acc / l_i
    tl.store(
        out_ptr + token_idx * stride_ot + head_idx * stride_oh + offs_d * stride_od,
        out,
        mask=offs_d < head_dim,
    )


def attention_paged_varlen_triton(
    query,
    key_cache,
    value_cache,
    query_lens,
    kv_lens,
    block_table,
):
    """Triton implementation of suffix-aligned paged varlen attention.

    The interface requires each query sequence to correspond to the final
    `q_len` positions of the KV sequence. Query token `j` therefore attends
    only through KV position `kv_len - q_len + j`.
    """
    block_size, max_blocks_per_seq = _validate_paged_attention_inputs(
        query, key_cache, value_cache, query_lens, kv_lens, block_table
    )
    query_seq_ids, query_offsets = build_query_metadata(query_lens)
    total_q, num_heads, head_dim = query.shape
    max_kv_len = int(kv_lens.max().item())
    block_d = triton.next_power_of_2(head_dim)
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(query)

    grid = (total_q * num_heads,)
    _paged_varlen_kernel[grid](
        query,
        key_cache,
        value_cache,
        out,
        query_seq_ids,
        query_offsets,
        query_lens,
        kv_lens,
        block_table,
        total_q,
        num_heads,
        head_dim,
        block_size,
        max_blocks_per_seq,
        max_kv_len,
        block_table.stride(0),
        block_table.stride(1),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        scale,
        BLOCK_D=block_d,
    )
    return out


def test_attention_paged_varlen_matches_reference(device):
    case = make_paged_inputs_case_one()
    out = attention_paged_varlen_triton(*case)
    ref = attention_paged_varlen_reference(*case)
    torch.testing.assert_close(out, ref, atol=5e-4, rtol=5e-4)


def test_attention_paged_varlen_partial_block(device):
    case = make_paged_inputs_case_two()
    out = attention_paged_varlen_triton(*case)
    ref = attention_paged_varlen_reference(*case)
    torch.testing.assert_close(out, ref, atol=5e-4, rtol=5e-4)

    query, key_cache, value_cache, query_lens, kv_lens, block_table = case
    padded_block_table = torch.full((2, 4), -1, dtype=torch.int32)
    padded_block_table[:, ::2] = block_table
    noncontig_block_table = padded_block_table[:, ::2]
    assert not noncontig_block_table.is_contiguous()
    noncontig_out = attention_paged_varlen_triton(
        query, key_cache, value_cache, query_lens, kv_lens, noncontig_block_table
    )
    noncontig_ref = attention_paged_varlen_reference(
        query, key_cache, value_cache, query_lens, kv_lens, noncontig_block_table
    )
    torch.testing.assert_close(noncontig_out, noncontig_ref, atol=5e-4, rtol=5e-4)

    with pytest.raises(ValueError):
        attention_paged_varlen_triton(
            query,
            key_cache,
            value_cache,
            query_lens,
            kv_lens,
            block_table[:, :1],
        )
