import torch

from .cross_attention import cross_attention


def test_cross_attention():
    torch.manual_seed(0)
    query = torch.randn(2, 3, 4)
    key = torch.randn(2, 5, 4)
    value = torch.randn(2, 5, 4)
    torch.testing.assert_close(
        cross_attention(query, key, value),
        torch.nn.functional.scaled_dot_product_attention(query, key, value),
        rtol=1e-4,
        atol=1e-4,
    )
