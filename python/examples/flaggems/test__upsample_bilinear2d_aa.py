import torch
import torch.nn.functional as F

from ._upsample_bilinear2d_aa import _upsample_bilinear2d_aa


def test_upsample_bilinear2d_aa_identity_size():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 4, 4, dtype=torch.float32, device="cpu")

    out = _upsample_bilinear2d_aa(x, (4, 4), align_corners=False)
    ref = F.interpolate(
        x, size=(4, 4), mode="bilinear", align_corners=False, antialias=True
    )

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_upsample_bilinear2d_aa_downsample_with_scale_factors():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 6, 8, dtype=torch.float32, device="cpu")

    out = _upsample_bilinear2d_aa(
        x,
        (3, 4),
        align_corners=False,
        scales_h=0.5,
        scales_w=0.5,
    )
    ref = F.interpolate(
        x,
        scale_factor=(0.5, 0.5),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_upsample_bilinear2d_aa_downsample_mixed_scales():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7, 11, dtype=torch.float32, device="cpu")

    out = _upsample_bilinear2d_aa(x, (5, 4), align_corners=False)
    ref = F.interpolate(
        x, size=(5, 4), mode="bilinear", align_corners=False, antialias=True
    )

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
