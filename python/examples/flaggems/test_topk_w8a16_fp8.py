import torch

from .topk_w8a16_fp8 import topk_w8a16_fp8


def test_topk_w8a16_fp8():
    values = torch.tensor([[0.1, 0.4, 0.2, 0.3]])
    got_v, got_i = topk_w8a16_fp8(values, 2)
    ref_v, ref_i = torch.topk(values, 2)
    torch.testing.assert_close(got_v, ref_v, rtol=1e-4, atol=1e-4)
    assert torch.equal(got_i, ref_i)
