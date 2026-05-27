import pytest
import torch

from .flash_api import mha_fwd


@pytest.mark.parametrize(
    "batch_size, seqlen_q, num_heads, head_dim",
    [
        (1, 32, 2, 32),
        (1, 64, 2, 32),
        (2, 32, 2, 32),
    ],
)
def test_mha_fwd_small(batch_size, seqlen_q, num_heads, head_dim):
    """Test mha_fwd forward pass on small shapes with fp16."""
    torch.manual_seed(0)
    q = torch.randn(
        batch_size,
        seqlen_q,
        num_heads,
        head_dim,
        dtype=torch.float16,
    )
    k = torch.randn(
        batch_size,
        seqlen_q,
        num_heads,
        head_dim,
        dtype=torch.float16,
    )
    v = torch.randn(
        batch_size,
        seqlen_q,
        num_heads,
        head_dim,
        dtype=torch.float16,
    )

    # removed: CUDA guard for triton-riscv

    softmax_scale = 1.0 / (head_dim**0.5)
    out, _, _, _, _, _, _, _ = mha_fwd(
        q,
        k,
        v,
        out=None,
        alibi_slopes=None,
        p_dropout=0.0,
        softmax_scale=softmax_scale,
        is_causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        return_softmax=False,
    )

    assert out.shape == q.shape, (
        f"Expected output shape {q.shape}, got {out.shape}"
    )
    assert not torch.isnan(out).any(), "Output contains NaN values"
    assert not torch.isinf(out).any(), "Output contains Inf values"


@pytest.mark.parametrize("total_q", [512, 1024])
def test_mha_fwd_medium(total_q):
    """Test mha_fwd with medium sizes (total_q tokens)."""
    torch.manual_seed(0)
    batch_size = 1
    num_heads = 2
    head_dim = 32
    seqlen_q = total_q

    q = torch.randn(
        batch_size,
        seqlen_q,
        num_heads,
        head_dim,
        dtype=torch.float16,
    )
    k = torch.randn(
        batch_size,
        seqlen_q,
        num_heads,
        head_dim,
        dtype=torch.float16,
    )
    v = torch.randn(
        batch_size,
        seqlen_q,
        num_heads,
        head_dim,
        dtype=torch.float16,
    )

    # removed: CUDA guard for triton-riscv

    softmax_scale = 1.0 / (head_dim**0.5)
    out, _, _, _, _, _, _, _ = mha_fwd(
        q,
        k,
        v,
        out=None,
        alibi_slopes=None,
        p_dropout=0.0,
        softmax_scale=softmax_scale,
        is_causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        return_softmax=False,
    )

    assert out.shape == (batch_size, seqlen_q, num_heads, head_dim), (
        f"Expected ({batch_size}, {seqlen_q}, {num_heads}, {head_dim}), got {out.shape}"
    )
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert out.dtype == torch.float16, f"Expected float16, got {out.dtype}"


def test_mha_fwd_compile():
    """Verify mha_fwd kernel compiles and produces valid output.

    This is a compilation smoke test: creates fp16 tensors on CPU,
    skips if no GPU is available (FlashAttention requires CUDA).
    """
    torch.manual_seed(0)
    q = torch.randn(1, 16, 2, 32, dtype=torch.float16)
    k = torch.randn(1, 16, 2, 32, dtype=torch.float16)
    v = torch.randn(1, 16, 2, 32, dtype=torch.float16)

    # removed: CUDA guard for triton-riscv

    softmax_scale = 1.0 / (32**0.5)
    out, _, _, _, _, _, _, _ = mha_fwd(
        q,
        k,
        v,
        out=None,
        alibi_slopes=None,
        p_dropout=0.0,
        softmax_scale=softmax_scale,
        is_causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        return_softmax=False,
    )

    assert out is not None
    assert out.shape == q.shape
    assert out.dtype == torch.float16
