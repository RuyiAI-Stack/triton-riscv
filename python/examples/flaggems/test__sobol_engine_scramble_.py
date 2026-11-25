import torch

from ._sobol_engine_scramble_ import _sobol_engine_scramble_


def test__sobol_engine_scramble_():
    state = torch.zeros(2, 4, dtype=torch.int32)
    scrambled = _sobol_engine_scramble_(state, torch.ones_like(state))
    assert scrambled.shape == state.shape
