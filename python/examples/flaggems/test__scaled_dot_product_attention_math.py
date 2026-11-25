import torch

from ._scaled_dot_product_attention_math import _scaled_dot_product_attention_math


def test__scaled_dot_product_attention_math():
    torch.manual_seed(0)
    query = torch.randn(1, 2, 4)
    torch.testing.assert_close(
        _scaled_dot_product_attention_math(query, query, query),
        torch.nn.functional.scaled_dot_product_attention(query, query, query),
        rtol=1e-4,
        atol=1e-4,
    )
