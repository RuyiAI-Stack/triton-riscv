import pytest
import torch

from .get_scheduler_metadata import get_scheduler_metadata


@pytest.mark.parametrize(
    "batch_size, max_seqlen_q, max_seqlen_k, num_heads, num_heads_k, headdim, headdim_v",
    [
        (2, 128, 128, 8, 8, 64, 64),
        (4, 256, 256, 8, 4, 128, 128),
        (2, 512, 512, 16, 16, 64, 64),
        (1, 1024, 1024, 8, 8, 128, 128),
    ],
)
def test_get_scheduler_metadata(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    num_heads_k,
    headdim,
    headdim_v,
):
    torch.manual_seed(0)
    device = "cpu"
    dtype = torch.float16

    seqused_k = torch.full(
        (batch_size,), max_seqlen_k, dtype=torch.int32, device=device
    )

    metadata = get_scheduler_metadata(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        headdim=headdim,
        headdim_v=headdim_v,
        qkv_dtype=dtype,
        seqused_k=seqused_k,
    )

    assert isinstance(metadata, torch.Tensor)
    assert metadata.dtype == torch.int32
    assert metadata.device.type == device
    # metadata should have at least 1 entry (semaphore)
    assert metadata.numel() >= 1


@pytest.mark.parametrize(
    "batch_size, max_seqlen_q, max_seqlen_k, num_heads, num_heads_k, headdim, headdim_v",
    [
        (2, 128, 128, 8, 8, 64, 64),
        (4, 256, 256, 8, 4, 128, 128),
    ],
)
def test_get_scheduler_metadata_causal(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    num_heads_k,
    headdim,
    headdim_v,
):
    torch.manual_seed(0)
    device = "cpu"
    dtype = torch.float16

    seqused_k = torch.full(
        (batch_size,), max_seqlen_k, dtype=torch.int32, device=device
    )

    metadata = get_scheduler_metadata(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        headdim=headdim,
        headdim_v=headdim_v,
        qkv_dtype=dtype,
        seqused_k=seqused_k,
        is_causal=True,
    )

    assert isinstance(metadata, torch.Tensor)
    assert metadata.dtype == torch.int32
    assert metadata.device.type == device
    assert metadata.numel() >= 1


@pytest.mark.parametrize(
    "shape",
    [
        512,
        1023,
        1024,
    ],
)
def test_get_scheduler_metadata_varying_seqlen(shape):
    torch.manual_seed(0)
    device = "cpu"
    dtype = torch.float16

    batch_size = 2
    seqused_k = torch.full(
        (batch_size,), shape, dtype=torch.int32, device=device
    )

    metadata = get_scheduler_metadata(
        batch_size=batch_size,
        max_seqlen_q=shape,
        max_seqlen_k=shape,
        num_heads=8,
        num_heads_k=8,
        headdim=64,
        headdim_v=64,
        qkv_dtype=dtype,
        seqused_k=seqused_k,
    )

    assert isinstance(metadata, torch.Tensor)
    assert metadata.dtype == torch.int32
    assert metadata.device.type == device
