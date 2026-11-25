import torch

from .flash_attention_backward import flash_attention_backward


def test_flash_attention_backward():
    torch.manual_seed(0)
    query = torch.randn(1, 2, 4, requires_grad=True)
    key = torch.randn(1, 2, 4, requires_grad=True)
    value = torch.randn(1, 2, 4, requires_grad=True)
    out = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    out.sum().backward()
    dq, dk, dv = flash_attention_backward(
        torch.ones_like(out), query.detach(), key.detach(), value.detach()
    )
    torch.testing.assert_close(dq, query.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dk, key.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dv, value.grad, rtol=1e-4, atol=1e-4)
