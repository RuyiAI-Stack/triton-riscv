import torch

from .atleast_1d import atleast_1d
from .atleast_3d import atleast_3d


def test_atleast_1d_3d_consistency_with_torch_module():
    for x in [torch.tensor(2.0), torch.arange(5), torch.arange(6).reshape(2, 3)]:
        torch.testing.assert_close(atleast_1d(x), torch.atleast_1d(x), rtol=0, atol=0)
        torch.testing.assert_close(atleast_3d(x), torch.atleast_3d(x), rtol=0, atol=0)
        assert atleast_1d(x).shape == torch.atleast_1d(x).shape
        assert atleast_3d(x).shape == torch.atleast_3d(x).shape


def test_sequence_forms_match_torch():
    ts = [torch.tensor(1.0), torch.arange(3)]
    ours = atleast_3d(ts)
    ref = torch.atleast_3d(ts)
    assert len(ours) == len(ref)
    for o, r in zip(ours, ref):
        assert o.shape == r.shape
