import torch

from ._native_multi_head_attention import _native_multi_head_attention


def test__native_multi_head_attention():
    torch.manual_seed(0)
    embed, heads, length, batch = 8, 2, 3, 2
    query = torch.randn(batch, length, embed)
    weight = torch.randn(3 * embed, embed)
    bias = torch.randn(3 * embed)
    proj_w = torch.eye(embed)
    proj_b = torch.zeros(embed)
    out = _native_multi_head_attention(
        query, query, query, embed, heads, weight, bias, proj_w, proj_b
    )
    assert out.shape == query.shape and torch.isfinite(out).all()
