import pytest
import torch

from .multinomial import multinomial


@pytest.mark.parametrize("n_samples", [1, 5])
def test_multinomial_with_replacement(n_samples):
    torch.manual_seed(0)
    prob = torch.tensor(
        [0.1, 0.2, 0.3, 0.4], dtype=torch.float32, device="cpu"
    )

    tri_out = multinomial(prob, n_samples, with_replacement=True)

    assert tri_out.shape == (n_samples,)
    assert tri_out.dtype == torch.int64
    # All values should be valid indices
    assert (tri_out >= 0).all()
    assert (tri_out < 4).all()


@pytest.mark.parametrize("n_samples", [1, 3])
def test_multinomial_without_replacement(n_samples):
    torch.manual_seed(0)
    prob = torch.tensor(
        [0.1, 0.2, 0.3, 0.4], dtype=torch.float32, device="cpu"
    )

    ref_out = torch.multinomial(prob, n_samples, replacement=False)
    tri_out = multinomial(prob, n_samples, with_replacement=False)

    assert tri_out.shape == ref_out.shape
    assert tri_out.dtype == torch.int64
    assert (tri_out >= 0).all()
    assert (tri_out < 4).all()
