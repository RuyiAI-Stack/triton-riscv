import pytest
import torch

from .dropout import dropout, dropout_backward


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize("p", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_dropout(shape, p):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device="cpu", requires_grad=True)

    # Run triton implementation
    out, mask = dropout(x, p, train=True)

    # Check correctness of masking and scaling
    # Values are either 0 or scaled by 1/(1-p)
    if p == 1.0:
        assert torch.all(out == 0)
        assert torch.all(~mask)
    elif p == 0.0:
        torch.testing.assert_close(out, x)
        assert torch.all(mask)
    else:
        # Check scaling factor
        scale = 1.0 / (1.0 - p)
        # Verify values match where mask is true
        torch.testing.assert_close(out[mask], x[mask] * scale)
        # Verify values are zero where mask is false
        assert torch.all(out[~mask] == 0)

        # Verify that roughly (1-p) proportion of elements are kept
        # Using a wide tolerance due to randomness
        prop_kept = mask.float().mean().item()
        assert abs(prop_kept - (1.0 - p)) < 0.05

    # Check backward
    grad_out = torch.ones_like(out)
    grad_in = dropout_backward(grad_out, mask, 1.0 / (1.0 - p) if p < 1.0 else 0.0)

    if p == 1.0:
        assert torch.all(grad_in == 0)
    elif p == 0.0:
        torch.testing.assert_close(grad_in, grad_out)
    else:
        torch.testing.assert_close(grad_in[mask], grad_out[mask] * scale)
        assert torch.all(grad_in[~mask] == 0)


def test_dropout_repeated_calls_advance_philox_offset():
    torch.manual_seed(0)
    x = torch.ones((4096,), dtype=torch.float32, device="cpu")

    _, first_mask = dropout(x, 0.5, train=True)
    _, second_mask = dropout(x, 0.5, train=True)

    assert not torch.equal(first_mask, second_mask)
