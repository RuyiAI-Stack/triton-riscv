import torch
import torch.nn.functional as F

from .upsample_trilinear3d import upsample_trilinear3d


def test_upsample_trilinear3d():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 3, 4, 5, dtype=torch.float32, device="cpu")

    out = upsample_trilinear3d(x, (4, 5, 6), align_corners=False)
    ref = F.interpolate(x, size=(4, 5, 6), mode="trilinear", align_corners=False)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
