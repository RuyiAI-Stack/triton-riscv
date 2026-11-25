import torch

from ._sobol_engine_ff_ import _sobol_engine_ff_


def test__sobol_engine_ff_():
    state = torch.arange(6, dtype=torch.int32).view(2, 3)
    assert torch.equal(_sobol_engine_ff_(state), state)
