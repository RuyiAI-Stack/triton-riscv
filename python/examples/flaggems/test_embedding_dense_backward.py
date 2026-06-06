import pytest
import torch

from .embedding_dense_backward import embedding_dense_backward


@pytest.mark.parametrize(
    "num_embeddings, embedding_dim",
    [
        (100, 32),
        (50, 128),
        # Test required size handling around 512, 1024
        (1024, 64),
    ],
)
@pytest.mark.parametrize("shape", [(10,), (4, 16), (2, 3, 4)])
@pytest.mark.parametrize("padding_idx", [None, 0, 5])
@pytest.mark.parametrize("scale_grad_by_freq", [False, True])
def test_embedding_dense_backward(
    num_embeddings, embedding_dim, shape, padding_idx, scale_grad_by_freq
):
    torch.manual_seed(0)
    device = "cpu"

    indices = torch.randint(0, num_embeddings, shape, dtype=torch.int64, device=device)
    grad_output = torch.randn(
        (*shape, embedding_dim), dtype=torch.float32, device=device
    )

    # Use PyTorch to get reference output
    weight = torch.randn(
        num_embeddings,
        embedding_dim,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    out_ref = torch.nn.functional.embedding(
        indices,
        weight,
        padding_idx=padding_idx,
        scale_grad_by_freq=scale_grad_by_freq,
    )
    out_ref.backward(grad_output)

    ref_grad_weight = weight.grad

    tri_grad_weight = embedding_dense_backward(
        grad_output,
        indices,
        num_embeddings,
        padding_idx=padding_idx if padding_idx is not None else -1,
        scale_grad_by_freq=scale_grad_by_freq,
    )

    torch.testing.assert_close(tri_grad_weight, ref_grad_weight, rtol=1e-3, atol=1e-3)
