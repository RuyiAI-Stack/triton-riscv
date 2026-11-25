import torch

from .underscore_sobol_engine_draw import underscore_sobol_engine_draw


def test_underscore_sobol_engine_draw():
    points, _quasi = underscore_sobol_engine_draw(None, 4, None, 1, 0)
    assert points.shape == (4, 1) and torch.all((points >= 0) & (points <= 1))
