import torch

from .bernoulli import bernoulli


def test_bernoulli_probability_edges():
    zeros = torch.zeros(1024, dtype=torch.float32, device="cpu")
    ones = torch.ones(1024, dtype=torch.float32, device="cpu")

    torch.testing.assert_close(bernoulli(zeros), zeros)
    torch.testing.assert_close(bernoulli(ones), ones)
