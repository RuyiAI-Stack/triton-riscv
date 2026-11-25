import torch

from ._saturate_weight_to_fp16 import _saturate_weight_to_fp16


def test__saturate_weight_to_fp16():
    values = torch.tensor([1e6, -1e6, 1.0])
    torch.testing.assert_close(
        _saturate_weight_to_fp16(values),
        torch.tensor([65504.0, -65504.0, 1.0]),
        rtol=1e-4,
        atol=1e-4,
    )
