import torch
import torch.nn.functional as F

from .threshold_ import threshold_


def test_threshold_inplace():
    x = torch.tensor([-2.0, -0.5, 0.0, 1.0], dtype=torch.float32, device="cpu")
    ref = F.threshold(x, threshold=0.0, value=-9.0)

    ret = threshold_(x, 0.0, -9.0)

    assert ret is x
    torch.testing.assert_close(x, ref)
