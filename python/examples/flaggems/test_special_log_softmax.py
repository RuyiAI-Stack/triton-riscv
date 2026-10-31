import torch
import torch.nn.functional as F

from .special_log_softmax import special_log_softmax


def test_special_log_softmax():
    torch.manual_seed(0)
    x = torch.randn(4, 16, dtype=torch.float32, device="cpu")

    out = special_log_softmax(x, dim=1)
    ref = F.log_softmax(x, dim=1)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_special_log_softmax_non_last_dim():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, dtype=torch.float32, device="cpu")

    out = special_log_softmax(x, dim=1)
    ref = F.log_softmax(x, dim=1)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
