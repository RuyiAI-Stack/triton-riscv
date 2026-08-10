import pytest
import torch

from .embedding import embedding, embedding_backward


@pytest.mark.parametrize(
    "num_embeddings, embedding_dim",
    [
        (100, 32),
        (50, 128),
        # Test required size handling around 512, 1024
        (512, 64),
        (1023, 64),
        (1024, 64),
    ],
)
@pytest.mark.parametrize("shape", [(10,), (4, 16), (2, 3, 4)])
@pytest.mark.parametrize("padding_idx", [None, 0, 5])
@pytest.mark.parametrize("scale_grad_by_freq", [False, True])
def test_embedding(
    num_embeddings, embedding_dim, shape, padding_idx, scale_grad_by_freq
):
    torch.manual_seed(0)
    device = "cpu"

    weight = torch.randn(
        num_embeddings,
        embedding_dim,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    indices = torch.randint(0, num_embeddings, shape, dtype=torch.int64, device=device)

    weight_ref = weight.clone().detach().requires_grad_(True)

    tri_out = embedding(
        weight,
        indices,
        padding_idx=padding_idx if padding_idx is not None else -1,
        scale_grad_by_freq=scale_grad_by_freq,
    )

    ref_out = torch.nn.functional.embedding(
        indices,
        weight_ref,
        padding_idx=padding_idx,
        scale_grad_by_freq=scale_grad_by_freq,
    )

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)

    grad_out = torch.randn_like(tri_out)

    # Backward using Triton
    grad_in = embedding_backward(
        grad_out,
        indices,
        num_embeddings,
        padding_idx=padding_idx if padding_idx is not None else -1,
        scale_grad_by_freq=scale_grad_by_freq,
    )

    # Backward using PyTorch
    ref_out.backward(grad_out)

    torch.testing.assert_close(grad_in, weight_ref.grad, rtol=1e-3, atol=1e-3)
