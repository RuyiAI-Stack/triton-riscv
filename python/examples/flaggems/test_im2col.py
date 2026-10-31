import torch
import torch.nn.functional as F

from .im2col import im2col


def test_im2col_matches_unfold():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 6, dtype=torch.float32, device="cpu")

    out = im2col(x, kernel_size=(3, 2), dilation=1, padding=(1, 0), stride=(2, 1))
    ref = F.unfold(x, kernel_size=(3, 2), dilation=1, padding=(1, 0), stride=(2, 1))

    torch.testing.assert_close(out, ref)
